import argparse, math, subprocess
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ---------------------- helpers ----------------------

def snap_to_patch_multiple(size, patch):
    return max(patch, (size // patch) * patch)

def resize_for_compute(frame_bgr, target_mpx, patch):
    """
    Downscale frame for feature compute (to avoid OOM) but keep aspect ratio.
    Returns (compute_frame, scale_w, scale_h).
    If target_mpx <= 0, returns original frame and (1.0, 1.0).
    """
    H, W = frame_bgr.shape[:2]
    if target_mpx <= 0:
        return frame_bgr, 1.0, 1.0
    aspect = W / H
    target_pixels = int(target_mpx * 1_000_000)
    newH = int(round((target_pixels / aspect) ** 0.5))
    newW = int(round(newH * aspect))
    # snap to patch multiples for ViT
    newH = snap_to_patch_multiple(newH, patch)
    newW = snap_to_patch_multiple(newW, patch)
    # clamp to original bounds
    newH = max(patch, min(newH, H))
    newW = max(patch, min(newW, W))
    if newH == H and newW == W:
        return frame_bgr, 1.0, 1.0
    resized = cv2.resize(frame_bgr, (newW, newH), interpolation=cv2.INTER_AREA)
    return resized, newW / W, newH / H

def extract_patches(frame_bgr, model, processor, device, dtype, do_resize=False):
    """
    Return (patches [N,C], (gH,gW)) for the given frame size.
    do_resize=False keeps the provided size (no processor-side resizing).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt", do_resize=do_resize)
    moved = {k: (v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device))
             for k, v in inputs.items()}
    with torch.inference_mode():
        last_hidden = model(**moved).last_hidden_state[0].detach().cpu().numpy()  # [tokens, C]
    # Drop CLS + any DINOv3 register tokens
    num_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    patches = last_hidden[1 + num_reg:]  # [N_patches, C]
    # Derive grid from tensor size and patch size
    _, _, H, W = moved["pixel_values"].shape
    patch = int(getattr(model.config, "patch_size", 16))
    gH, gW = H // patch, W // patch
    if gH * gW != len(patches):  # rare mismatch; fall back to square
        g = int(round(math.sqrt(len(patches))))
        gH, gW = g, g
    return patches, (gH, gW)

def cosine_sim(a, b, eps=1e-8):
    """a: [Na,C], b: [Nb,C] -> [Na,Nb]"""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return a @ b.T

def open_ffmpeg_writer(W, H, fps, out_path, crf=18, preset="slow"):
    """
    High-quality H.264 writer via FFmpeg pipe.
    Returns (proc, write_fn, close_fn).
    """
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
        proc.stdin.close()
        proc.wait()
    return proc, write_fn, close_fn

# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path (.mp4/.mov)")
    ap.add_argument("--model", default="facebook/dinov2-base",
                    help="HF model id or local dir (use DINOv3 id/path if you have access)")
    ap.add_argument("--fps", type=float, default=10.0, help="Sampling FPS and output FPS")
    ap.add_argument("--out", default="pseudotrack_path.mp4", help="Output video path")
    ap.add_argument("--half", action="store_true", help="Use float16 on mps/cuda")
    ap.add_argument("--ffmpeg", action="store_true", help="Use FFmpeg for high-quality encoding")
    ap.add_argument("--alpha", type=float, default=0.45, help="Heatmap overlay alpha (0..1)")
    ap.add_argument("--conf", type=float, default=0.70, help="Min similarity to accept a new position")
    ap.add_argument("--ema", type=float, default=0.20, help="EMA smoothing factor for dot updates (0..1)")
    ap.add_argument("--ref_blend", type=float, default=0.05, help="Slow reference adaptation (0 disables)")
    ap.add_argument("--duration", type=float, default=0.0, help="Process only N seconds (0 = full video)")
    ap.add_argument("--compute_mpx", type=float, default=0.7,
                    help="Megapixels for feature compute; 0 = full-res (4K will likely OOM)")
    ap.add_argument("--line_thickness", type=int, default=2, help="Red path line thickness (px)")
    ap.add_argument("--dot_radius", type=int, default=5, help="Red dot radius (px)")
    args = ap.parse_args()

    # Device & dtype
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half and device in ("mps", "cuda") else torch.float32
    print(f"[info] device={device} dtype={dtype} model={args.model}")

    # Model
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=dtype).eval().to(device)
    patch = int(getattr(model.config, "patch_size", 16))

    # Video
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

    max_frames = int(round(args.duration * src_fps)) if args.duration > 0 else None

    # Grab first sampled frame for the click
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
        cap.release()
        raise RuntimeError("No sampled frames (check --fps / --duration).")

    # Compute at downscaled size to avoid OOM, then map back to orig coords
    comp0, sw0, sh0 = resize_for_compute(first_frame, args.compute_mpx, patch)
    first_patches, (gH0, gW0) = extract_patches(comp0, model, processor, device, dtype, do_resize=False)

    # UI: click to choose focus (on original-res preview)
    ref_pt = {"pt": None}
    win = "Click a point to focus (Enter/Space to start, q/ESC to cancel)"
    def on_mouse(ev,x,y,flags,param):
        if ev == cv2.EVENT_LBUTTONDOWN:
            ref_pt["pt"] = (x,y)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL); cv2.resizeWindow(win, W, H)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        draw = first_frame.copy()
        if ref_pt["pt"] is not None:
            cv2.circle(draw, ref_pt["pt"], args.dot_radius, (0,0,255), -1, cv2.LINE_AA)  # red preview dot
        cv2.imshow(win, draw)
        k = cv2.waitKey(10) & 0xFF
        if ref_pt["pt"] is not None and k in (13, 32, ord('s')):  # Enter/Space/S
            break
        if k in (27, ord('q')):  # ESC/q
            cap.release(); cv2.destroyAllWindows(); print("[info] cancelled."); return
    cv2.destroyWindow(win)

    # Build reference feature from clicked patch (unit-normalized)
    x0, y0 = ref_pt["pt"]
    x0c, y0c = x0 * sw0, y0 * sh0
    gx0 = int(np.clip(x0c / comp0.shape[1] * gW0, 0, gW0 - 1))
    gy0 = int(np.clip(y0c / comp0.shape[0] * gH0, 0, gH0 - 1))
    ref_vec = first_patches[gy0 * gW0 + gx0 : gy0 * gW0 + gx0 + 1].copy()
    ref_vec = ref_vec / (np.linalg.norm(ref_vec, axis=-1, keepdims=True) + 1e-8)

    # Tracking state
    dot_x, dot_y = float(x0), float(y0)
    path_points = [(int(round(dot_x)), int(round(dot_y)))]

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

    # Rewind and process frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if max_frames is not None and frame_idx >= max_frames: break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        # Downscale for compute, then map results back to original geometry
        comp, sw, sh = resize_for_compute(frame, args.compute_mpx, patch)
        patches, (gH, gW) = extract_patches(comp, model, processor, device, dtype, do_resize=False)

        sim = cosine_sim(patches, ref_vec).reshape(gH, gW)
        smin, smax = float(sim.min()), float(sim.max())
        sim_n = (sim - smin) / (smax - smin + 1e-8)

        # Peak in compute grid -> pixel in original frame
        peak_idx = int(np.argmax(sim))
        py, px = divmod(peak_idx, gW)
        cell_w_c, cell_h_c = comp.shape[1] / gW, comp.shape[0] / gH
        peak_x_c, peak_y_c = (px + 0.5) * cell_w_c, (py + 0.5) * cell_h_c
        peak_x, peak_y = peak_x_c / sw, peak_y_c / sh

        # Confidence gate + EMA smoothing
        peak_conf = float(sim_n.max())  # 0..1
        if peak_conf >= args.conf:
            dot_x = (1.0 - args.ema) * dot_x + args.ema * peak_x
            dot_y = (1.0 - args.ema) * dot_y + args.ema * peak_y
            # optional slow adaptation of the reference feature
            if args.ref_blend > 0.0:
                peak_feat = patches[py * gW + px : py * gW + px + 1]
                peak_feat = peak_feat / (np.linalg.norm(peak_feat, axis=-1, keepdims=True) + 1e-8)
                a = args.ref_blend
                ref_vec = (1 - a) * ref_vec + a * peak_feat
                ref_vec = ref_vec / (np.linalg.norm(ref_vec, axis=-1, keepdims=True) + 1e-8)

        # Append to red path
        path_points.append((int(round(dot_x)), int(round(dot_y))))

        # Heatmap overlay (upsample to original size for visualization)
        sim_up = cv2.resize(sim_n, (W, H), interpolation=cv2.INTER_CUBIC)
        heat = cv2.applyColorMap((sim_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
        out = cv2.addWeighted(frame, 1.0 - args.alpha, heat, args.alpha, 0)

        # Draw red path and current red dot
        if len(path_points) >= 2:
            cv2.polylines(out, [np.array(path_points, dtype=np.int32)], False,
                          (0, 0, 255), args.line_thickness, cv2.LINE_AA)
        cv2.circle(out, (int(round(dot_x)), int(round(dot_y))),
                   args.dot_radius, (0, 0, 255), -1, cv2.LINE_AA)

        # Write frame
        if writer is not None: writer.write(out)
        else: write_fn(out)

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
