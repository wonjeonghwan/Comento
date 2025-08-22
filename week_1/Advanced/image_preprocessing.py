from pathlib import Path
from typing import Tuple
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
EDGE_DIFF_THRESHOLD: int = 120          # 에지 강도 임계값
MIN_EDGE_RATIO: float = 0.02            # 강한 에지 비율 최소치

MAX_TO_SAVE: int = 5                    # 저장할 샘플 수



def resize_to_target(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return ImageOps.fit(img.convert("RGB"), size, method=Image.BICUBIC)

def to_grayscale(img_rgb: Image.Image) -> Image.Image:
    return ImageOps.grayscale(img_rgb)

def blur(img: Image.Image, radius: float = GAUSSIAN_BLUR_RADIUS) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))

def hflip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)

# -------------------------------------------------------------------
# 이상치 필터
# -------------------------------------------------------------------
def is_too_dark(img_gray_u8: Image.Image, mean_thr: float = DARK_MEAN_THRESHOLD) -> bool:
    return np.array(img_gray_u8).mean() < mean_thr

def is_too_small_proxy(img_gray_u8: Image.Image,
                       edge_thr: int = EDGE_DIFF_THRESHOLD,
                       min_edge_ratio: float = MIN_EDGE_RATIO) -> bool:
    arr = np.array(img_gray_u8, dtype=np.int16)
    gy = np.abs(arr[:, 1:] - arr[:, :-1])
    gx = np.abs(arr[1:, :] - arr[:-1, :])
    strong_edges = (gx > edge_thr).sum() + (gy > edge_thr).sum()
    edge_ratio = strong_edges / (arr.shape[0] * arr.shape[1])
    return edge_ratio < min_edge_ratio

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
