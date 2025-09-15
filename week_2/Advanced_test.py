import os
import cv2
import numpy as np
import pytest
from Advanced import load_image, to_gray, make_depth_map, to_point_cloud


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
