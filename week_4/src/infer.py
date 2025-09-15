import argparse
from PIL import Image
from pathlib import Path
import torch
from torchvision import transforms

from models.conv_ae import ConvAE
from utils.common import load_threshold, collect_images, device_auto

def make_tf(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ])

def mse_err(x, xrec):
    return ((x - xrec) ** 2).mean(dim=(1,2,3)).item()

SCRIPT_DIR = Path(__file__).resolve().parent   # week_4/src
ROOT = SCRIPT_DIR.parent                       # week_4
DEFAULT_MODEL = (ROOT / "outputs" / "ae_best.pt").resolve()
DEFAULT_THR   = (ROOT / "outputs" / "threshold.json").resolve()
DEFAULT_PATH  = (ROOT / "data" / "새 폴더").resolve()

def main():
    ap = argparse.ArgumentParser()
    
    
    
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--threshold", type=str, default=str(DEFAULT_THR))
    ap.add_argument("--path", type=str, default=str(DEFAULT_PATH))  # ← required 제거 + 기본값 지정
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=128)
    args = ap.parse_args()

    device = device_auto()
    thr = load_threshold(args.threshold)

    model = ConvAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    tf = make_tf(args.img_size)
    paths = collect_images(args.path)

    print(f"[Scoring] N={len(paths)}  threshold={thr:.6f}")
    for p in paths:
        img = Image.open(p).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            xrec = model(x)
        e = mse_err(x, xrec)
        label = "ANOMALY" if e > thr else "NORMAL"
        print(f"{p.name:40s}  err={e:.6f}  => {label}")

if __name__ == "__main__":
    main()
