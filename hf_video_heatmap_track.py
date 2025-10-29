import argparse, math, cv2, torch, time, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# python hf_video_heatmap_track.py --video driving.mov --model facebook/dinov3-vits16-pretrain-lvd1689m --fps 10 --scale_mpx 0 --out driving_heatmap2.mp4 --preview
# -------------------- Utils --------------------
def pick_device(use_half: bool):
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    dtype = torch.float16 if use_half and dev in ("cuda","mps") else torch.float32
    return dev, dtype

def load_model(model_id_or_dir, device, dtype):
    proc = AutoImageProcessor.from_pretrained(model_id_or_dir)
    model = AutoModel.from_pretrained(model_id_or_dir, dtype=dtype).eval().to(device)
    return proc, model

def get_mean_std(proc, dtype=torch.float32):
    mean = getattr(proc, "image_mean", [0.485, 0.456, 0.406])
    std  = getattr(proc, "image_std",  [0.229, 0.224, 0.225])
    return (torch.tensor(mean, dtype=dtype).view(3,1,1),
            torch.tensor(std,  dtype=dtype).view(3,1,1))

def resize_to_megapixels(frame_bgr, target_mpx: float, patch: int):
    """Resize frame so H*W ~= target_mpx*1e6 and both dims are multiples of patch."""
    H, W = frame_bgr.shape[:2]
    if target_mpx <= 0:  # no resize
        newW, newH = W, H
    else:
        aspect = W / H
        target_pixels = max(1, int(target_mpx * 1_000_000))
        newH = int(round((target_pixels / aspect) ** 0.5))
        newW = int(round(newH * aspect))
    # snap to patch multiples
    newH = max(patch, (newH // patch) * patch)
    newW = max(patch, (newW // patch) * patch)
    if newH == H and newW == W:
        return frame_bgr
    return cv2.resize(frame_bgr, (newW, newH), interpolation=cv2.INTER_AREA)

def extract_patches(proc, model, bgr, device, dtype):
    """Return (patches [N,C], disp_bgr [H,W,3], gridHW (gH,gW))."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = proc(images=pil, return_tensors="pt")
    moved = {k: (v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device)) for k,v in inputs.items()}

    with torch.inference_mode():
        last_hidden = model(**moved).last_hidden_state[0].detach().cpu()  # [tokens, C]

    num_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    feats = last_hidden.numpy()
    patches = feats[1 + num_reg :]  # drop CLS (+ registers)

    _, _, H, W = moved["pixel_values"].shape   # [1,3,H,W]
    patch = int(getattr(model.config, "patch_size", 16))
    gH, gW = H // patch, W // patch
    if gH * gW != len(patches):  # fallback if any mismatch
        g = int(round(math.sqrt(len(patches))))
        gH, gW = g, g

    # reconstruct display image (un-normalize)
    pix = moved["pixel_values"][0].detach().cpu()  # [3,H,W] normalized
    mean, std = get_mean_std(proc, dtype=pix.dtype)
    disp = (pix * std + mean).clamp(0,1).permute(1,2,0).numpy()
    disp_bgr = (disp * 255).astype("uint8")[:, :, ::-1]
    return patches, disp_bgr, (gH, gW)

def cosine_sim(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a @ b.T  # [Na, Nb]

# -------------------- Main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="input video path")
    ap.add_argument("--out", default="tracked_heatmap.mp4", help="output mp4 path")
    ap.add_argument("--model", default="facebook/dinov2-base",
                    help="HF model id or local dir (use your DINOv3 id/path if you have access)")
    ap.add_argument("--fps", type=float, default=20.0, help="sampling FPS (frames per second)")
    ap.add_argument("--scale_mpx", type=float, default=0.5,
                    help="target megapixels for model input (0 = no resize). 0.5 ~ 0.5MP")
    ap.add_argument("--half", action="store_true", help="use float16 on mps/cuda")
    ap.add_argument("--alpha", type=float, default=0.45, help="heatmap overlay alpha (0..1)")
    ap.add_argument("--preview", action="store_true", help="show live preview window while processing")
    args = ap.parse_args()

    device, dtype = pick_device(args.half)
    print(f"[info] device={device} dtype={dtype} model={args.model}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # Read metadata
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    src_W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / args.fps)))  # sample frames
    print(f"[info] source: {src_W}x{src_H}@{src_fps:.2f}fps, total {total_src} frames, sampling every {step} frame(s)")

    # Load model once
    proc, model = load_model(args.model, device, dtype)
    patch = int(getattr(model.config, "patch_size", 16))

    # Grab first sampled frame and prepare user click
    first_idx = None
    first_disp = None
    first_patches = None
    first_grid = None
    frame_i = -1
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % step != 0:
            continue
        # resize for model compute
        bgr_resized = resize_to_megapixels(bgr, args.scale_mpx, patch)
        patches, disp_bgr, (gH, gW) = extract_patches(proc, model, bgr_resized, device, dtype)
        first_idx = frame_i
        first_disp = disp_bgr
        first_patches = patches
        first_grid = (gH, gW)
        break
    if first_disp is None:
        cap.release()
        raise RuntimeError("No frames sampled from the video (check --fps).")

    H, W = first_disp.shape[:2]

    # Ask for click on the first processed frame
    click_pt = {"pt": None}
    def on_mouse(ev,x,y,flags,param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            click_pt["pt"] = (x,y)

    win = "Click a point to start tracking (q/ESC to cancel)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        show = first_disp.copy()
        if click_pt["pt"] is not None:
            cv2.circle(show, click_pt["pt"], 6, (0,255,0), 2)
        cv2.imshow(win, show)
        k = cv2.waitKey(10) & 0xFF
        if click_pt["pt"] is not None and k in (13, 32, ord('s'), ord('S')):  # Enter/Space/S to start
            break
        if k in (27, ord('q')):  # ESC or q to abort
            cv2.destroyWindow(win)
            cap.release()
            print("[info] cancelled.")
            return
    cv2.destroyWindow(win)

    # Compute reference patch from first frame click
    gH, gW = first_grid
    x, y = click_pt["pt"]
    gx = int(np.clip(x / W * gW, 0, gW-1))
    gy = int(np.clip(y / H * gH, 0, gH-1))
    ref = first_patches[gy*gW + gx : gy*gW + gx + 1]

    # Prepare writer (use processed frame size)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter (codec/permissions).")

    # Optionally preview window
    if args.preview:
        pv = "Tracking preview (press q/ESC to stop preview; video will still save)"
        cv2.namedWindow(pv, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(pv, W, H)

    # Rewind to start and process all sampled frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_i = -1
    processed = 0
    t0 = time.time()

    def render_heatmap(patches, disp_bgr, gridHW):
        gH, gW = gridHW
        H, W = disp_bgr.shape[:2]
        sim = cosine_sim(patches, ref).reshape(gH, gW)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        sim_up = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap((sim_up*255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(disp_bgr, 1 - args.alpha, heat, args.alpha, 0)
        # mark the reference click in the same (first frame) coordinates; for later frames we just draw the same pixel
        cv2.circle(out, (x,y), 6, (0,255,0), 2)
        return out

    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frame_i += 1
        if frame_i % step != 0:
            continue

        bgr_resized = resize_to_megapixels(bgr, args.scale_mpx, patch)
        patches, disp_bgr, gridHW = extract_patches(proc, model, bgr_resized, device, dtype)
        out = render_heatmap(patches, disp_bgr, gridHW)
        writer.write(out)
        processed += 1

        if args.preview:
            cv2.imshow(pv, out)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                # stop preview but keep writing the file
                cv2.destroyWindow(pv)
                args.preview = False

        # simple progress print
        if processed % 10 == 0:
            elapsed = time.time() - t0
            fps_eff = processed / max(1e-6, elapsed)
            print(f"[info] processed {processed} frames at ~{fps_eff:.2f} fps")

    writer.release()
    cap.release()
    if args.preview:
        cv2.destroyAllWindows()

    print(f"[done] wrote {processed} frames to {args.out}")

if __name__ == "__main__":
    main()
