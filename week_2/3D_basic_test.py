import os
import cv2
import numpy as np
import pytest

def load_image():
    """sample.jpg 로드 (BGR)"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "sample.jpg")
    image = cv2.imread(image_path)  # 전제: sample.jpg 존재/정상
    return image

def to_gray(image):
    """그레이스케일 변환"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def make_depth_map(gray):
    """COLORMAP_JET 적용"""
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return depth_map


# Unit test
def test_load_image():
    img = load_image()
    
    assert isinstance(img, np.ndarray), "cv2.imread 결과가 ndarray가 아님"
    assert img.ndim == 3 and img.shape[2] == 3, "BGR 3채널 이미지가 아님"
    assert img.dtype == np.uint8, "이미지 dtype은 uint8이어야 함"

def test_to_gray():
    img = load_image()
    gray = to_gray(img)
    
    assert isinstance(gray, np.ndarray), "그레이스케일 결과가 ndarray가 아님"
    assert gray.ndim == 2, "그레이스케일은 2차원이어야 함"
    assert gray.shape == img.shape[:2], "크기가 원본 HxW와 달라서는 안 됨"
    assert gray.dtype == np.uint8, "그레이스케일 dtype은 uint8이어야 함"

def test_make_depth_map():
    img = load_image()
    gray = to_gray(img)
    depth = make_depth_map(gray)
    
    assert isinstance(depth, np.ndarray), "깊이맵 결과가 ndarray가 아님"
    assert depth.ndim == 3 and depth.shape[2] == 3, "깊이맵은 3채널 컬러여야 함"
    assert depth.shape[:2] == gray.shape, "깊이맵 HxW는 그레이와 동일해야 함"
    assert depth.dtype == np.uint8, "깊이맵 dtype은 uint8이어야 함"


if __name__ == "__main__":
    pytest.main()