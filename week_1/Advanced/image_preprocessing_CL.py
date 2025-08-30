from pathlib import Path
from typing import Tuple
import cv2
import random
import numpy as np
import argparse
from datasets import load_dataset
from PIL import Image, ImageFilter, ImageOps


# 커맨드 라인 정의
def parse_args():
    parser = argparse.ArgumentParser(description="Image Preprocessing Script")
    parser.add_argument("--save_dir", type=str, default="preprocessed_output",
                        help="이미지 저장 경로 (default: preprocessed_output)")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                        help="리사이즈 타겟 크기 (default: 224 224)")
    parser.add_argument("--blur_radius", type=float, default=0.2,
                        help="가우시안 블러 반경 (default: 0.2)")
    parser.add_argument("--dark_thr", type=float, default=40.0,
                        help="어두움 판별 임계값 (default: 40.0)")
    parser.add_argument("--min_area_ratio", type=float, default=0.005,
                        help="객체 최소 면적 비율 (default: 0.005)")
    parser.add_argument("--max_to_save", type=int, default=5,
                        help="저장할 이미지 샘플 수 (default: 5)")
    return parser.parse_args()


# 실행 함수
def resize_to_target(img, size: Tuple[int, int]):
    return ImageOps.fit(img.convert("RGB"), size, method=Image.BICUBIC)

def to_grayscale(img_rgb):
    return ImageOps.grayscale(img_rgb)

def blur(img, radius: float):
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def hflip(img):
    return ImageOps.mirror(img)

def is_too_dark(img_gray_u8, mean_thr: float):
    return np.array(img_gray_u8).mean() < mean_thr

def is_too_small_proxy(img_gray_u8, min_area_ratio: float):
    arr = np.array(img_gray_u8, dtype=np.uint8)
    edges = cv2.Canny(arr, 50, 150)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return True
    h, w = arr.shape[:2]
    img_area = float(h * w)
    max_area = max(cv2.contourArea(c) for c in cnts)
    return (max_area / img_area) < min_area_ratio

def save_image(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)

# 메인 실행
def process_and_save(args):
    BASE_DIR = Path(__file__).resolve().parent
    SAVE_DIR = BASE_DIR / args.save_dir
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    STAGE_DIRS = {
        "orig": SAVE_DIR / "1_original",
        "resized": SAVE_DIR / "2_resized",
        "gray": SAVE_DIR / "3_grayscale",
        "blur": SAVE_DIR / "4_blurred",
        "aug": SAVE_DIR / "5_augmented",
    }
    for p in STAGE_DIRS.values():
        p.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ethz/food101", split="train")
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    saved = 0
    for i in idxs:
        sample = ds[i]
        orig_pil = sample["image"]
        label = sample["label"]
        fname = f"idx{i}_label{label}.jpg"

        # 1) Original
        save_image(orig_pil.convert("RGB"), STAGE_DIRS["orig"] / fname)

        # 2) Resize
        resized_rgb = resize_to_target(orig_pil, tuple(args.target_size))
        save_image(resized_rgb, STAGE_DIRS["resized"] / fname)

        # 3) Grayscale
        gray = to_grayscale(resized_rgb)
        save_image(gray, STAGE_DIRS["gray"] / fname)

        # * 이상치 필터링
        if is_too_dark(gray, args.dark_thr):
            continue
        if is_too_small_proxy(gray, args.min_area_ratio):
            continue

        # 4) Blur
        denoised = blur(gray, args.blur_radius)
        save_image(denoised, STAGE_DIRS["blur"] / fname)

        # 5) Augment
        augmented = hflip(denoised)
        save_image(augmented, STAGE_DIRS["aug"] / fname)

        saved += 1
        if saved >= args.max_to_save:
            break

    print(f"[Done] Saved {saved} images → {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    args = parse_args()
    process_and_save(args)