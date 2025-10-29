import argparse, os, math, cv2, torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ---------- Utils ----------
def pick_device(prefer_half: bool):
    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"
    if prefer_half and dev in ("cuda", "mps"):
        # mps prefers float16; cuda supports fp16/bf16 (we'll use fp16)
        dtype = torch.float16
    else:
        dtype = torch.float32
    return dev, dtype

def iter_video_frames(path, target_fps=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / target_fps)) if target_fps else 1, 1)
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, frame_bgr
        i += 1
    cap.release()

def cosine_sim(a: np.ndarray, b: np.ndarray, eps=1e-8):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a @ b.T  # [N_a, N_b]

def heatmap_overlay(img_bgr, heat, alpha=0.5):
    # heat expected in [0,1]; convert to color and overlay
    heat_uint8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(img_bgr, 1 - alpha, heat_color, alpha, 0)
    return out

def grid_shape_from_tokens(n_tokens):
    g = int(round(math.sqrt(n_tokens)))
    assert g * g == n_tokens, f"Patch tokens ({n_tokens}) are not a square grid"
    return g, g

# ---------- Model wrapper ----------
class DinoV3Extractor:
    def __init__(self, model_id, device, dtype):
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
        self.model.eval().to(device)
        self.device = device
        self.dtype = dtype

    @torch.inference_mode()
    def extract(self, frame_bgr):
        # resize to model’s expected size via processor; keep a resized frame for display
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = self.processor(images=pil, return_tensors="pt")
        # push to device/dtype
        inputs = {k: (v.to(self.device, dtype=self.dtype) if v.dtype.is_floating_point else v.to(self.device))
                  for k, v in inputs.items()}
        out = self.model(**inputs)
        feats = out.last_hidden_state[0].detach().cpu().numpy()  # [tokens, C]
        cls = feats[0]
        patches = feats[1:]  # [N_patches, C]
        # also keep the model input image (for consistent display)
        img_proc = self.processor.post_process(images=[inputs["pixel_values"][0].detach().cpu()], do_rescale=True)[0]
        img_disp = (img_proc.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # HxWx3 RGB
        img_disp_bgr = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
        return cls, patches, img_disp_bgr

# ---------- Click handler ----------
class ClickPicker:
    def __init__(self, win_name, grid_hw):
        self.win = win_name
        self.grid_h, self.grid_w = grid_hw
        self.clicked = False
        self.xy = None
        cv2.setMouseCallback(self.win, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = True
            self.xy = (x, y)

    def pick_patch_index(self, img_shape_hw):
        # map clicked pixel in display image to patch index in grid
        if not self.clicked or self.xy is None:
            return None
        H, W = img_shape_hw
        px, py = self.xy
        # clamp
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)
        gx = int(px / W * self.grid_w)
        gy = int(py / H * self.grid_h)
        gx = np.clip(gx, 0, self.grid_w - 1)
        gy = np.clip(gy, 0, self.grid_h - 1)
        return gy * self.grid_w + gx  # row-major

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--fps", type=float, default=20.0, help="sample FPS for processing/vis")
    ap.add_argument("--half", action="store_true", help="use float16 if supported (mps/cuda)")
    ap.add_argument("--save", default="", help="optional path to save an mp4 (e.g., out.mp4)")
    ap.add_argument("--alpha", type=float, default=0.45, help="heatmap overlay alpha")
    args = ap.parse_args()

    device, dtype = pick_device(args.half)
    print(f"[info] device={device} dtype={dtype}")

    extractor = DinoV3Extractor(args.model, device, dtype)

    # Pull first frame to select a reference patch by clicking
    frames = iter_video_frames(args.video, target_fps=args.fps)
    try:
        idx0, bgr0 = next(frames)
    except StopIteration:
        raise RuntimeError("Empty or unreadable video")

    cls0, patches0, disp0 = extractor.extract(bgr0)
    gH, gW = grid_shape_from_tokens(len(patches0))
    print(f"[info] first frame patches grid = {gH}x{gW}, dim={patches0.shape[1]}")

    win = "DINOv3 Track Viz (click a point, press any key to start)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp0.shape[1], disp0.shape[0])
    picker = ClickPicker(win, (gH, gW))

    # show first frame until user clicks
    while True:
        show = disp0.copy()
        if picker.xy:
            cv2.circle(show, picker.xy, 6, (0, 255, 0), 2)
        cv2.imshow(win, show)
        key = cv2.waitKey(10)
        if picker.clicked and key != -1:
            break

    ref_idx = picker.pick_patch_index((disp0.shape[0], disp0.shape[1]))
    if ref_idx is None:
        cv2.destroyAllWindows()
        raise RuntimeError("No point selected.")
    ref_vec = patches0[ref_idx:ref_idx + 1]  # [1, C]
    print(f"[info] reference patch index: {ref_idx}")

    # optional writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, args.fps, (disp0.shape[1], disp0.shape[0]))

    # show first frame with a tiny heatmap spike at the selected patch
    sim0 = cosine_sim(patches0, ref_vec).reshape(gH, gW)  # [H,W]
    sim0 = (sim0 - sim0.min()) / (sim0.max() - sim0.min() + 1e-8)
    sim0_up = cv2.resize(sim0, (disp0.shape[1], disp0.shape[0]), interpolation=cv2.INTER_CUBIC)
    out0 = heatmap_overlay(disp0, sim0_up, alpha=args.alpha)
    cv2.imshow(win, out0)
    if writer: writer.write(out0)
    cv2.waitKey(1)

    # process the rest
    for idx, bgr in frames:
        _, patches, disp = extractor.extract(bgr)
        # recompute grid each time in case model changes resolution (shouldn’t)
        gH, gW = grid_shape_from_tokens(len(patches))
        sim = cosine_sim(patches, ref_vec).reshape(gH, gW)
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)

        sim_up = cv2.resize(sim, (disp.shape[1], disp.shape[0]), interpolation=cv2.INTER_CUBIC)
        out = heatmap_overlay(disp, sim_up, alpha=args.alpha)

        # visualize the clicked point location for reference
        if picker.xy:
            cv2.circle(out, picker.xy, 6, (0, 255, 0), 2)

        cv2.imshow(win, out)
        if writer: writer.write(out)

        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break

    if writer:
        writer.release()
        print(f"[info] saved visualization to {args.save}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
