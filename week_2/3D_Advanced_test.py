import os
import cv2
import numpy as np
import pytest


# 이미지 로드
def load_image():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "sample.jpg")
    image = cv2.imread(image_path)
    return image

# 그레이스케일 변환
def to_gray(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Depth Map 생성
def make_depth_map(gray: np.ndarray):
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return depth_map

# 3D 포인트 클라우드 변환
def to_point_cloud(depth_map: np.ndarray, gray):
    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)  # 깊이값을 Z 축으로 사용
    points_3d = np.dstack((X, Y, Z))  # (H, W, 3)
    return points_3d


#  === Unit Test ===
def test_load_image():
    img = load_image()
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3 and img.shape[2] == 3
    assert img.dtype == np.uint8

def test_to_gray():
    img = load_image()
    gray = to_gray(img)
    assert gray.ndim == 2
    assert gray.shape == img.shape[:2]

def test_make_depth_map():
    img = load_image()
    gray = to_gray(img)
    depth = make_depth_map(gray)
    assert depth.shape[:2] == gray.shape
    assert depth.ndim == 3 and depth.shape[2] == 3

def test_to_point_cloud():
    img = load_image()
    gray = to_gray(img)
    depth = make_depth_map(gray)
    points = to_point_cloud(depth, gray)
    assert points.shape[:2] == gray.shape
    assert points.shape[2] == 3

# ===== pytest 실행 진입점 =====
if __name__ == "__main__":
    pytest.main()
