import argparse, math, subprocess
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

def extract_patches(frame_bgr, model, processor, device, dtype):
    """Return (patches [N,C], gridHW (gH,gW)). No resize; native resolution."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt", do_resize=False)
    moved = {k: (v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device))
             for k, v in inputs.items()}
    with torch.inference_mode():
        last_hidden = model(**moved).last_hidden_state[0].detach().cpu().numpy()  # [tokens, C]
    num_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    patches = last_hidden[1 + num_reg:]  # drop CLS + registers
    _, _, H, W = moved["pixel_values"].shape  # [1,3,H,W]
    patch = int(getattr(model.config, "patch_size", 16))
    gH, gW = H // patch, W // patch
    if gH * gW != len(patches):  # fallback if something mismatched
        g = int(round(math.sqrt(len(patches))))
        gH, gW = g, g
    return patches, (gH, gW)

def cosine_sim(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a @ b.T  # [Na, Nb]

def open_ffmpeg_writer(W, H, fps, out_path, crf=18, preset="slow"):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s:v", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    def write_fn(frame: np.ndarray):
        proc.stdin.write(frame.tobytes())
    def close_fn():
        proc.stdin.close()
        proc.wait()
    return proc, write_fn, close_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--model", default="facebook/dinov2-base",
                    help="HF model id or local dir (DINOv3 id/path if you have access)")
    ap.add_argument("--fps", type=float, default=10.0, help="Sampling FPS and output FPS")
    ap.add_argument("--out", default="tracked_heatmap.mp4", help="Output video path")
    ap.add_argument("--half", action="store_true", help="Use float16 on mps/cuda")
    ap.add_argument("--ffmpeg", action="store_true", help="Use FFmpeg for high-quality encoding")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half and device in ("mps","cuda") else torch.float32
    print(f"[info] device={device} dtype={dtype} model={args.model}")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=dtype).eval().to(device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] video: {W}x{H} @ {src_fps:.2f} fps, total {total} frames")

    # Sampling stride
    step = max(1, int(round(src_fps / max(1e-6, args.fps))))
    print(f"[info] sampling every {step} frame(s) → output {args.fps} fps")

    # Grab the first sampled frame to pick a reference click
    frame_idx = -1
    first_frame = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        first_frame = frame.copy()
        break
    if first_frame is None:
        cap.release()
        raise RuntimeError("No sampled frames (check --fps).")

    # Compute patches for the first frame (for picking a ref patch)
    first_patches, (gH, gW) = extract_patches(first_frame, model, processor, device, dtype)

    # Click to select focus (on original-res first frame)
    ref_pt = {"pt": None}
    win = "Click a point to focus (press Enter/Space to start, q/ESC to cancel)"
    def on_mouse(ev,x,y,flags,param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            ref_pt["pt"] = (x,y)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        draw = first_frame.copy()
        if ref_pt["pt"] is not None:
            cv2.circle(draw, ref_pt["pt"], 6, (0,255,0), 2)
        cv2.imshow(win, draw)
        k = cv2.waitKey(10) & 0xFF
        if ref_pt["pt"] is not None and k in (13, 32, ord('s')):  # Enter/Space/S
            break
        if k in (27, ord('q')):  # ESC/q
            cap.release(); cv2.destroyAllWindows()
            print("[info] cancelled.")
            return
    cv2.destroyWindow(win)

    # Map pixel -> patch index, build ref vector
    x, y = ref_pt["pt"]
    gx = int(np.clip(x / W * gW, 0, gW-1))
    gy = int(np.clip(y / H * gH, 0, gH-1))
    ref = first_patches[gy*gW + gx : gy*gW + gx + 1]  # [1,C]

    # Writer (FFmpeg high quality or OpenCV fallback)
    writer = None
    write_fn = close_fn = None
    if args.ffmpeg:
        try:
            _, write_fn, close_fn = open_ffmpeg_writer(W, H, args.fps, args.out, crf=18, preset="slow")
            print("[info] using FFmpeg pipe")
        except FileNotFoundError:
            print("[warn] ffmpeg not found; falling back to OpenCV")
    if write_fn is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out, fourcc, args.fps, (W, H))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Failed to open OpenCV VideoWriter")

    # Rewind and process all sampled frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        patches, (gH, gW) = extract_patches(frame, model, processor, device, dtype)
        sim = cosine_sim(patches, ref).reshape(gH, gW)  # similarity to chosen patch
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        sim_up = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC)

        heat = cv2.applyColorMap((sim_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)
        # draw the same reference pixel for visualization
        cv2.circle(out, (x,y), 6, (0,255,0), 2)

        if writer is not None:
            writer.write(out)
        else:
            write_fn(out)

        written += 1
        if written % 10 == 0:
            print(f"[info] wrote {written} frames")

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    elif close_fn is not None:
        close_fn()
    print(f"[done] saved {written} frames to {args.out}")

if __name__ == "__main__":
    main()
