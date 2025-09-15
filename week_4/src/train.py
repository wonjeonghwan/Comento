import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from data.dataset import ImageFolderNoLabel
from models.conv_ae import ConvAE
from utils.common import set_seed, reconstruction_error, save_threshold, device_auto

# -------------------------
# RGBA → RGB(흰 배경) 일관화
# -------------------------
def rgba_to_rgb_white(img: Image.Image) -> Image.Image:
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")

def make_train_tf(img_size: int):
    return transforms.Compose([
        transforms.Lambda(rgba_to_rgb_white),
        # 크기/위치 류 (오토인코더 특성상 과한 왜곡은 금지)
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        # 색상 류 (만화풍이므로 살짝만)
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        # 텐서에서 동작. 너무 세게 주지 않음
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0.0),
    ])

def make_val_tf(img_size: int):
    return transforms.Compose([
        transforms.Lambda(rgba_to_rgb_white),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="week_4/data/피카츄")
    ap.add_argument("--out_dir",  type=str, default="week_4/outputs")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)   # 소량 데이터에 맞춰 축소
    ap.add_argument("--epochs", type=int, default=150)      # EarlyStop으로 멈추게 길게
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--seed", type=int, default=42)
    # Early Stopping
    ap.add_argument("--patience", type=int, default=12, help="no improvement epochs to wait before stop")
    ap.add_argument("--min_delta", type=float, default=1e-4, help="minimum improvement in val loss to reset patience")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    device = device_auto()

    # -------------------------
    # Dataset & Split
    # -------------------------
    # 주: ImageFolderNoLabel이 tf 인자를 받도록 되어 있어야 함.
    full_clean = ImageFolderNoLabel(args.data_dir, img_size=args.img_size, aug=False, tf=make_val_tf(args.img_size))
    n = len(full_clean)
    idx = np.arange(n); np.random.shuffle(idx)
    cut = int(n * 0.9)
    tr_idx, va_idx = idx[:cut], idx[cut:]

    # 학습: 강한 증강 파이프라인
    train_base = ImageFolderNoLabel(args.data_dir, img_size=args.img_size, aug=False, tf=make_train_tf(args.img_size))
    train_ds = Subset(train_base, tr_idx)

    # 검증/임계값: 증강 없는 파이프라인 (clean)
    val_base = full_clean
    val_ds = Subset(val_base, va_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------
    # Model / Optim
    # -------------------------
    model = ConvAE(latent_dim=args.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    cri = nn.MSELoss()

    best = float("inf")
    ckpt = os.path.join(args.out_dir, "ae_best.pt")
    thr_path = os.path.join(args.out_dir, "threshold.json")
    no_improve = 0

    # -------------------------
    # Train
    # -------------------------
    for ep in range(1, args.epochs + 1):
        model.train(); tr_loss = 0.0
        for x in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}"):
            x = x.to(device)
            xrec = model(x)
            loss = cri(xrec, x)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * x.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval(); va_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                xrec = model(x)
                va_loss += cri(xrec, x).item() * x.size(0)
        va_loss /= len(val_loader.dataset)
        sched.step()

        print(f"[EP {ep}] train_mse={tr_loss:.6f}  val_mse={va_loss:.6f}")

        if (best - va_loss) > args.min_delta:
            best = va_loss
            no_improve = 0
            torch.save(model.state_dict(), ckpt)
            print(f"  ↳ Improved! best_val_mse={best:.6f} (model saved)")
        else:
            no_improve += 1
            print(f"  ↳ No improvement ({no_improve}/{args.patience})")

        if no_improve >= args.patience:
            print(f"Early stopping triggered at epoch {ep}. Best val_mse={best:.6f}")
            break

    # -------------------------
    # Threshold from CLEAN set (val_ds)
    # -------------------------
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    cal_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    errs = []
    with torch.no_grad():
        for x in tqdm(cal_loader, desc="Calibrating threshold"):
            x = x.to(device)
            e = reconstruction_error(x, model(x)).cpu().numpy()
            errs.append(e)
    errs = np.concatenate(errs)
    meta = save_threshold(errs, args.percentile, thr_path)
    print(f"Saved: {ckpt}, {thr_path} -> {meta}")

if __name__ == "__main__":
    main()
