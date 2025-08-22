from pathlib import Path
from typing import Tuple
import cv2
import random
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageFilter, ImageOps

# 경로 지정
SAVE_DIR = Path("preprocessed_output")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# THRESHOLD
TARGET_SIZE: Tuple[int, int] = (224, 224)
GAUSSIAN_BLUR_RADIUS: float = 0.2       # 가우시안 블러 값
DARK_MEAN_THRESHOLD: float = 40.0       # 0~255 평균 밝기 임계값
MIN_AREA_RATIO: float = 0.005

MAX_TO_SAVE: int = 5                    # 저장할 샘플 수


# 기본문제 처리
def resize_to_target(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), size, method=Image.BICUBIC)

def to_grayscale(img_rgb: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img_rgb)

def blur(img: Image.Image, radius: float = GAUSSIAN_BLUR_RADIUS) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def hflip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


# 이상치 
def is_too_dark(img_gray_u8: Image.Image, mean_thr: float = DARK_MEAN_THRESHOLD) -> bool:
    return np.array(img_gray_u8).mean() < mean_thr

def is_too_small_proxy(img_gray_u8: Image.Image,
                        min_area_ratio: float = MIN_AREA_RATIO) -> bool:
    # 가장 큰 객체의 면적 / 전체 이미지 면적 < min_area_ratio 이면 '너무 작다'로 판정
    arr = np.array(img_gray_u8, dtype=np.uint8)

    # 엣지 검출
    edges = cv2.Canny(arr, 50, 150)

    # 면적 추출
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return True  # 윤곽이 전혀 없으면 '너무 작다/비어있다'로 처리

    # 가장 큰 컨투어 면적 비율
    h, w = arr.shape[:2]
    img_area = float(h * w)
    max_area = max(cv2.contourArea(c) for c in cnts)

    area_ratio = max_area / img_area
    return area_ratio < min_area_ratio

# 저장
def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


# 메인 실행
def process_and_save():
    
    ds = load_dataset("ethz/food101", split="train")
    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    saved = 0
    for i in idxs:
        sample = ds[i]
        orig_pil: Image.Image = sample["image"]
        label: int = sample["label"]

        # 1) Resize
        resized_rgb = resize_to_target(orig_pil, TARGET_SIZE)

        # 2) Grayscale
        gray = to_grayscale(resized_rgb)

        # 3) 이상치 필터 
        if is_too_dark(gray):
            continue
        if is_too_small_proxy(gray):
            continue

        # 4) Denoise(Blur)
        denoised = blur(gray)

        # 5) Augment(Horizontal Flip)
        processed = hflip(denoised)

        # 저장
        out_orig = SAVE_DIR / f"original_idx{i}_label{label}.jpg"
        out_proc = SAVE_DIR / f"processed_idx{i}_label{label}.jpg"

        save_image(orig_pil, out_orig)     # 시각화용 RGB
        save_image(processed,   out_proc)     # 시각화용 Gray (Blur+Flip)

        saved += 1
        if saved >= MAX_TO_SAVE:
            break

    print(f"[Done] Saved {saved} image pairs → {SAVE_DIR.resolve()}")


if __name__ == "__main__":
    process_and_save()
