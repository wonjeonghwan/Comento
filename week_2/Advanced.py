#### 심화 코드: Depth Map을 기반으로 3D 포인트 클라우드 생성
import cv2
import numpy as np
import os

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

# 결과 출력
if __name__ == "__main__":
    image = load_image()
    gray = to_gray(image)
    depth_map = make_depth_map(gray)        # 여기서 depth_map 생성
    points_3d = to_point_cloud(depth_map, gray)

    # 결과 출력
    cv2.imshow('Depth Map', depth_map)      # 이제 depth_map 존재함
    cv2.waitKey(0)
    cv2.destroyAllWindows()
