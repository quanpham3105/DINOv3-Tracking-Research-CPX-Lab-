import argparse, math, subprocess
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ---------------------- Helpers ----------------------

def extract_patches(frame_bgr, model, processor, device, dtype):
    """Return (patches [N,C], (gH,gW)) at native resolution (no resize)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt", do_resize=False)
    moved = {k: (v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device))
             for k, v in inputs.items()}
    with torch.inference_mode():
        last_hidden = model(**moved).last_hidden_state[0].detach().cpu().numpy()  # [tokens, C]
    num_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    patches = last_hidden[1 + num_reg:]  # drop CLS + registers
    _, _, H, W = moved["pixel_values"].shape
    patch = int(getattr(model.config, "patch_size", 16))
    gH, gW = H // patch, W // patch
    if gH * gW != len(patches):
        g = int(round(math.sqrt(len(patches))))
        gH, gW = g, g
    return patches, (gH, gW)

def cosine_sim(a, b, eps=1e-8):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a @ b.T

def open_ffmpeg_writer(W, H, fps, out_path, crf=18, preset="slow"):
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s:v", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p", out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    def write_fn(frame_bgr: np.ndarray):
        proc.stdin.write(frame_bgr.tobytes())
    def close_fn():
        proc.stdin.close(); proc.wait()
    return proc, write_fn, close_fn

# ---------------------- Main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path (.mp4/.mov)")
    ap.add_argument("--model", default="facebook/dinov2-base",
                    help="HF model id or local snapshot (use your DINOv3 id/path if you have access)")
    ap.add_argument("--fps", type=float, default=10.0, help="Sampling FPS and output FPS")
    ap.add_argument("--out", default="pseudotrack_path.mp4", help="Output video path")
    ap.add_argument("--half", action="store_true", help="Use float16 on mps/cuda")
    ap.add_argument("--ffmpeg", action="store_true", help="Use FFmpeg for high-quality encoding")
    ap.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha (0..1)")
    ap.add_argument("--conf", type=float, default=0.70, help="Min similarity to accept a new position")
    ap.add_argument("--ema", type=float, default=0.20, help="EMA smoothing factor for dot updates (0..1)")
    ap.add_argument("--ref_blend", type=float, default=0.05, help="Slow reference adaptation (0 disables)")
    ap.add_argument("--duration", type=float, default=0.0, help="Process only N seconds (0 = full video)")
    ap.add_argument("--line_thickness", type=int, default=2, help="Red path line thickness (px)")
    ap.add_argument("--dot_radius", type=int, default=4, help="Red dot radius (px)")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half and device in ("mps", "cuda") else torch.float32
    print(f"[info] device={device} dtype={dtype} model={args.model}")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=dtype).eval().to(device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[info] video: {W}x{H} @ {src_fps:.2f} fps, total {total} frames")

    step = max(1, int(round(src_fps / max(1e-6, args.fps))))
    print(f"[info] sampling every {step} frame(s) → output {args.fps} fps")

    max_frames = int(round(args.duration * src_fps)) if args.duration > 0 else None

    # Pick first sampled frame
    frame_idx = -1
    first_frame = None
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame_idx += 1
        if max_frames is not None and frame_idx >= max_frames: break
        if frame_idx % step != 0: continue
        first_frame = frame.copy()
        break
    if first_frame is None:
        cap.release(); raise RuntimeError("No sampled frames (check --fps / --duration).")

    # First-frame patches
    first_patches, (gH, gW) = extract_patches(first_frame, model, processor, device, dtype)

    # Click to choose focus
    ref_pt = {"pt": None}
    win = "Click a point to focus (Enter/Space to start, q/ESC to cancel)"
    def on_mouse(ev,x,y,flags,param):
        if ev == cv2.EVENT_LBUTTONDOWN: ref_pt["pt"] = (x,y)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, W, H)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        draw = first_frame.copy()
        if ref_pt["pt"] is not None:
            cv2.circle(draw, ref_pt["pt"], args.dot_radius, (0,0,255), -1, cv2.LINE_AA)  # red preview dot
        cv2.imshow(win, draw)
        k = cv2.waitKey(10) & 0xFF
        if ref_pt["pt"] is not None and k in (13, 32, ord('s')): break  # Enter/Space/S
        if k in (27, ord('q')): cap.release(); cv2.destroyAllWindows(); print("[info] cancelled."); return
    cv2.destroyWindow(win)

    # Reference feature (unit-normalized)
    x0, y0 = ref_pt["pt"]
    gx0 = int(np.clip(x0 / W * gW, 0, gW-1)); gy0 = int(np.clip(y0 / H * gH, 0, gH-1))
    ref_vec = first_patches[gy0 * gW + gx0 : gy0 * gW + gx0 + 1].copy()
    ref_vec = ref_vec / (np.linalg.norm(ref_vec, axis=-1, keepdims=True) + 1e-8)

    # Tracking state
    dot_x, dot_y = float(x0), float(y0)
    path_points = [ (int(dot_x), int(dot_y)) ]  # full trajectory (pixel coords)

    # Writer
    writer = None; write_fn = close_fn = None
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
            cap.release(); raise RuntimeError("Failed to open OpenCV VideoWriter")

    # Rewind and process all sampled frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if max_frames is not None and frame_idx >= max_frames: break
        if frame_idx % step != 0:
            frame_idx += 1; continue

        patches, (gH, gW) = extract_patches(frame, model, processor, device, dtype)

        # Similarity & normalize 0..1
        sim = cosine_sim(patches, ref_vec).reshape(gH, gW)
        smin, smax = float(sim.min()), float(sim.max())
        sim_n = (sim - smin) / (smax - smin + 1e-8)

        # Peak -> pixel
        peak_idx = int(np.argmax(sim))
        py, px = divmod(peak_idx, gW)
        cell_w, cell_h = W / gW, H / gH
        peak_x, peak_y = (px + 0.5) * cell_w, (py + 0.5) * cell_h

        # Confidence + EMA smoothing
        peak_conf = float(sim_n.max())
        if peak_conf >= args.conf:
            dot_x = (1.0 - args.ema) * dot_x + args.ema * peak_x
            dot_y = (1.0 - args.ema) * dot_y + args.ema * peak_y
            # optional slow adaptation
            if args.ref_blend > 0.0:
                peak_feat = patches[py * gW + px : py * gW + px + 1]
                peak_feat = peak_feat / (np.linalg.norm(peak_feat, axis=-1, keepdims=True) + 1e-8)
                a = args.ref_blend
                ref_vec = (1 - a) * ref_vec + a * peak_feat
                ref_vec = ref_vec / (np.linalg.norm(ref_vec, axis=-1, keepdims=True) + 1e-8)

        # Append current (smoothed) position to full path
        path_points.append((int(round(dot_x)), int(round(dot_y))))

        # Draw heatmap + red path
        sim_up = cv2.resize(sim_n, (W, H), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap((sim_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(frame, 1.0 - args.alpha, heat, args.alpha, 0)

        if len(path_points) >= 2:
            cv2.polylines(out, [np.array(path_points, dtype=np.int32)], False, (0,0,255), args.line_thickness, cv2.LINE_AA)
        # current point as red dot
        cv2.circle(out, (int(round(dot_x)), int(round(dot_y))), args.dot_radius, (0,0,255), -1, cv2.LINE_AA)

        if writer is not None: writer.write(out)
        else: write_fn(out)

        written += 1
        if written % 10 == 0:
            print(f"[info] wrote {written} frames")
        frame_idx += 1

    cap.release()
    if writer is not None: writer.release()
    elif close_fn is not None: close_fn()
    print(f"[done] saved {written} frames to {args.out}")

if __name__ == "__main__":
    main()
