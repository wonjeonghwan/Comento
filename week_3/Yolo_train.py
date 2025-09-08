from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="week_3/archive/data.yaml",
        epochs = 10,
        imgsz = 960,
        batch = 16,
        device = 0,
        
        # === 불/연기 특화 증강 ===
        # 기하 변형은 약하게(연기 박스가 쉽게 사라짐)
        degrees=3.0,        # 과도 회전은 금지
        translate=0.05,
        scale=0.30,         # ±30%
        shear=1.0,
        perspective=0.0,

        # 대칭성: 좌우 OK, 상하 금지(연기는 '위로' 퍼짐)
        fliplr=0.5,
        flipud=0.0,

        # 조명/색상: 야간, 연기·불 명암 편차 대응
        hsv_h=0.005,        # 불 색조 변형은 너무 크지 않게
        hsv_s=0.60,
        hsv_v=0.60,         # 명도 변형 강하게 → 저조도/역광 대응

        # 혼합 계열: 너무 세게 하면 연기 텍스처가 깨짐 → 보수적으로
        mosaic=0.35,        # 0.2~0.5 권장
        mixup=0.05,         # 0.0~0.1 사이
    )

    model.val()
    
if __name__ == "__main__":
    main()