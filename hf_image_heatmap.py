# hf_image_heatmap_live.py
import argparse, cv2, math, torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

def pick_device(use_half: bool):
    if torch.backends.mps.is_available(): dev = "mps"
    elif torch.cuda.is_available():        dev = "cuda"
    else:                                   dev = "cpu"
    dtype = torch.float16 if use_half and dev in ("cuda","mps") else torch.float32
    return dev, dtype

def load_model(model_id_or_dir, device, dtype):
    proc = AutoImageProcessor.from_pretrained(model_id_or_dir)
    model = AutoModel.from_pretrained(model_id_or_dir, dtype=dtype).eval().to(device)
    return proc, model

def get_mean_std(proc, dtype=torch.float32):
    mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
    std  = getattr(proc, "image_std",  [0.229, 0.224, 0.225])
    mean = torch.tensor(mean, dtype=dtype).view(3,1,1)
    std  = torch.tensor(std,  dtype=dtype).view(3,1,1)
    return mean, std

def extract_patches(proc, model, bgr, device, dtype):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = proc(images=pil, return_tensors="pt")
    moved = {k: (v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device)) for k,v in inputs.items()}
    with torch.inference_mode():
        last_hidden = model(**moved).last_hidden_state[0].detach().cpu()  # [tokens, dim]
    num_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    feats = last_hidden.numpy()
    patches = feats[1 + num_reg :]  # drop CLS (+ registers)
    _, _, H, W = moved["pixel_values"].shape
    patch = int(getattr(model.config, "patch_size", 16))
    gH, gW = H // patch, W // patch
    if gH * gW != len(patches):
        g = int(round(math.sqrt(len(patches)))); gH, gW = g, g
    pix = moved["pixel_values"][0].detach().cpu()  # [3,H,W] normalized
    mean, std = get_mean_std(proc, dtype=pix.dtype)
    disp = (pix * std + mean).clamp(0,1).permute(1,2,0).numpy()
    disp_bgr = (disp * 255).astype("uint8")[:, :, ::-1]
    return patches, disp_bgr, (gH, gW)

def cosine_sim(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a @ b.T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.45, help="heatmap overlay")
    ap.add_argument("--save", default="", help="optional: save last heatmap to this path on exit")
    args = ap.parse_args()

    device, dtype = pick_device(args.half)
    print(f"[info] device={device} dtype={dtype} model={args.model}")

    src_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if src_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    proc, model = load_model(args.model, device, dtype)
    patches, disp_bgr, (gH, gW) = extract_patches(proc, model, src_bgr, device, dtype)
    H, W = disp_bgr.shape[:2]
    win = "DINO heatmap (click to update; q/ESC to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)

    state = {"out": disp_bgr.copy()}  # keep last rendered frame

    def draw_heatmap(x, y):
        gx = int(np.clip(x / W * gW, 0, gW-1))
        gy = int(np.clip(y / H * gH, 0, gH-1))
        ref = patches[gy*gW + gx : gy*gW + gx + 1]
        sim = cosine_sim(patches, ref).reshape(gH, gW)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        sim_up = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap((sim_up*255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(disp_bgr, 1 - args.alpha, heat, args.alpha, 0)
        cv2.circle(out, (x, y), 6, (0,255,0), 2)
        state["out"] = out
        cv2.imshow(win, out)

    def on_mouse(ev, x, y, flags, param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            draw_heatmap(x, y)

    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, disp_bgr)

    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (27, ord('q')):  # ESC or q
            break

    if args.save:
        cv2.imwrite(args.save, state["out"])
        print(f"[info] saved last heatmap to {args.save}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
