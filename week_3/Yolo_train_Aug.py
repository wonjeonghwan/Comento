# Stage 1: 증강 켠 학습
from ultralytics import YOLO

def stage1():
    model = YOLO("yolov8n.pt")
    model.train(
        data="week_3/archive/data.yaml",
        epochs=8,
        imgsz=640,
        batch=16,
        device=0,
        degrees=3.0, translate=0.05, scale=0.30, shear=1.0,
        fliplr=0.5, flipud=0.0,
        hsv_h=0.005, hsv_s=0.60, hsv_v=0.60,
        mosaic=0.35, mixup=0.05, close_mosaic=10
    )

def stage2():  # 증강 OFF로 미세조정(원본 적합)
    model = YOLO("runs/detect/train/weights/last.pt")  # stage1 결과 사용
    model.train(
        data="week_3/archive/data.yaml",
        epochs=8,                # 총 epoch(추가 10 정도면 충분)
        imgsz=640,
        batch=16,
        device=0,
        # ▶ 증강 전부 끄기
        degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
        fliplr=0.0, flipud=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        mosaic=0.0, mixup=0.0, close_mosaic=0
    )

if __name__ == "__main__":
    # stage1()
    stage2()