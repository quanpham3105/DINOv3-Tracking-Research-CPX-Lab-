import argparse, os, cv2, torch, numpy as np
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

def iter_video_frames(path, target_fps=None):
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / target_fps)) if target_fps else 1, 1)
    i = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok: break
        if i % step == 0:
            yield i, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="output folder for npz files")
    ap.add_argument("--model", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    ap.add_argument("--fps", type=float, default=None, help="sample to this FPS (e.g., 20)")
    ap.add_argument("--half", action="store_true", help="use bfloat16/float16 if supported")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.half and torch.cuda.is_bf16_supported() else (torch.float16 if args.half and device=="cuda" else torch.float32)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype)
    model.eval().to(device)

    with torch.inference_mode():
        for idx, rgb in iter_video_frames(args.video, args.fps):
            image = Image.fromarray(rgb)
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device, dtype=dtype if v.dtype.is_floating_point else None) for k, v in inputs.items()}

            out = model(**inputs)
            # DINOv3 ViTs return:
            #  - last_hidden_state: [1, N_tokens, C] (CLS + patch tokens)
            #  - pooler_output (sometimes)
            feats = out.last_hidden_state[0].detach().cpu().numpy()  # [tokens, dim]
            # Split CLS vs patch tokens (CLS is token 0 for ViT)
            cls = feats[0]                      # [C]
            patches = feats[1:]                 # [N_patches, C]

            np.savez_compressed(
                os.path.join(args.out, f"frame_{idx:06d}.npz"),
                cls=cls,
                patches=patches,
                shape=np.array(image.size[::-1]),  # (H,W)
                model=args.model
            )

            # (optional) quick sanity print every ~1s of video
            if idx % 30 == 0:
                print(f"[frame {idx}] tokens={feats.shape[0]} dim={feats.shape[1]}")

if __name__ == "__main__":
    main()
