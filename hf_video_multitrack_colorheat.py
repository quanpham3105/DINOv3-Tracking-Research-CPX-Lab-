import argparse, math, subprocess
import cv2, torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


# ---------------- helpers ----------------

def snap_to_patch_multiple(size, patch):
    return max(patch, (size // patch) * patch)

def resize_for_compute(frame_bgr, target_mpx, patch):
    H, W = frame_bgr.shape[:2]
    if target_mpx <= 0:
        return frame_bgr, 1.0, 1.0
    aspect = W / H
    target_pixels = int(target_mpx * 1_000_000)
    newH = int(round((target_pixels / aspect) ** 0.5))
    newW = int(round(newH * aspect))
    newH = snap_to_patch_multiple(newH, patch)
    newW = snap_to_patch_multiple(newW, patch)
    newH = max(patch, min(newH, H))
    newW = max(patch, min(newW, W))
    if newH == H and newW == W:
        return frame_bgr, 1.0, 1.0
    resized = cv2.resize(frame_bgr, (newW, newH), interpolation=cv2.INTER_AREA)
    return resized, newW / W, newH / H


def extract_patches(frame_bgr, model, processor, device, dtype, do_resize=False):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = processor(images=pil, return_tensors="pt", do_resize=do_resize)
    moved = {k: (v.to(device, dtype=dtype) if v.dtype.is_floating_point else v.to(device))
             for k, v in inputs.items()}
    with torch.inference_mode():
        last_hidden = model(**moved).last_hidden_state[0].detach().cpu().numpy()
    num_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    patches = last_hidden[1 + num_reg:]
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
    def write_fn(frame_bgr):
        proc.stdin.write(frame_bgr.tobytes())
    def close_fn():
        proc.stdin.close(); proc.wait()
    return proc, write_fn, close_fn


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--out", default="multitrack_colorheat.mp4")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--ffmpeg", action="store_true")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--ema", type=float, default=0.2)
    ap.add_argument("--ref_blend", type=float, default=0.05)
    ap.add_argument("--compute_mpx", type=float, default=0.7)
    ap.add_argument("--line_thickness", type=int, default=2)
    ap.add_argument("--dot_radius", type=int, default=5)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half and device in ("mps","cuda") else torch.float32
    print(f"[info] device={device} dtype={dtype} model={args.model}")

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=dtype).eval().to(device)
    patch = int(getattr(model.config, "patch_size", 16))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError("Cannot open video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W,H = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(src_fps / args.fps)))
    print(f"[info] {W}x{H}@{src_fps:.1f}, every {step} frames")

    ok, first = cap.read()
    if not ok: raise RuntimeError("Failed to read first frame")

    comp0, sw0, sh0 = resize_for_compute(first, args.compute_mpx, patch)
    feats0, (gH0,gW0) = extract_patches(comp0, model, processor, device, dtype)

    # multiple click selection
    clicks = []
    win = "Click multiple targets (Enter to confirm)"
    def on_mouse(e,x,y,f,p):
        if e == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x,y))
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        disp = first.copy()
        for c in clicks:
            cv2.circle(disp, c, args.dot_radius, (0,0,255), -1, cv2.LINE_AA)
        cv2.imshow(win, disp)
        k = cv2.waitKey(10) & 0xFF
        if k in (13, 32): break
        if k in (27, ord('q')): return
    cv2.destroyWindow(win)

    trackers = []
    for (x0,y0) in clicks:
        x0c,y0c = x0*sw0,y0*sh0
        gx0 = int(np.clip(x0c/comp0.shape[1]*gW0,0,gW0-1))
        gy0 = int(np.clip(y0c/comp0.shape[0]*gH0,0,gH0-1))
        ref = feats0[gy0*gW0+gx0:gy0*gW0+gx0+1].copy()
        ref /= np.linalg.norm(ref,axis=-1,keepdims=True)+1e-8
        trackers.append({
            "ref": ref,
            "dot": [float(x0),float(y0)],
            "path":[(int(x0),int(y0))],
            "color": tuple(np.random.randint(64,255,3).tolist())
        })
    print(f"[info] tracking {len(trackers)} targets")

    # output writer
    if args.ffmpeg:
        _,write_fn,close_fn = open_ffmpeg_writer(W,H,args.fps,args.out)
    else:
        fourcc=cv2.VideoWriter_fourcc(*"mp4v")
        write_fn=None;close_fn=None
        writer=cv2.VideoWriter(args.out,fourcc,args.fps,(W,H))

    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    frame_idx,written=0,0
    while True:
        ok,frame=cap.read()
        if not ok:break
        if frame_idx%step!=0:
            frame_idx+=1;continue

        comp,sw,sh=resize_for_compute(frame,args.compute_mpx,patch)
        feats,(gH,gW)=extract_patches(comp,model,processor,device,dtype)

        # similarity per tracker
        sims=[]
        for t in trackers:
            sim=(cosine_sim(feats,t["ref"]).reshape(gH,gW))
            smin,smax=float(sim.min()),float(sim.max())
            sim=(sim-smin)/(smax-smin+1e-8)
            sims.append(sim)

            pk=int(np.argmax(sim));py,px=divmod(pk,gW)
            cw,ch=comp.shape[1]/gW,comp.shape[0]/gH
            px_c,py_c=(px+0.5)*cw,(py+0.5)*ch
            px_o,py_o=px_c/sw,py_c/sh
            conf=float(sim.max())
            if conf>=args.conf:
                t["dot"][0]=(1-args.ema)*t["dot"][0]+args.ema*px_o
                t["dot"][1]=(1-args.ema)*t["dot"][1]+args.ema*py_o
                if args.ref_blend>0:
                    pf=feats[py*gW+px:py*gW+px+1]
                    pf/=np.linalg.norm(pf,axis=-1,keepdims=True)+1e-8
                    a=args.ref_blend
                    t["ref"]=(1-a)*t["ref"]+a*pf
                    t["ref"]/=np.linalg.norm(t["ref"],axis=-1,keepdims=True)+1e-8
            t["path"].append((int(round(t["dot"][0])),int(round(t["dot"][1]))))

        # color-blended per-target heatmaps
        out = frame.copy()
        for sim, t in zip(sims, trackers):
            sim_up = cv2.resize(sim, (W, H), interpolation=cv2.INTER_CUBIC)
            color = np.array(t["color"], np.uint8)
            heat = np.zeros_like(out)
            for c in range(3):
                heat[:, :, c] = (sim_up * color[c]).astype(np.uint8)
            out = cv2.addWeighted(out, 1 - args.alpha, heat, args.alpha / len(trackers), 0)

        # draw dots & lines
        for t in trackers:
            if len(t["path"])>1:
                cv2.polylines(out,[np.array(t["path"],np.int32)],False,t["color"],args.line_thickness,cv2.LINE_AA)
            cv2.circle(out,(int(t["dot"][0]),int(t["dot"][1])),args.dot_radius,t["color"],-1,cv2.LINE_AA)

        if write_fn:write_fn(out)
        else:writer.write(out)
        written+=1
        if written%10==0:print(f"[info] {written} frames")
        frame_idx+=1

    cap.release()
    if write_fn:close_fn()
    else:writer.release()
    print(f"[done] saved {written} frames to {args.out}")


if __name__=="__main__":
    main()
