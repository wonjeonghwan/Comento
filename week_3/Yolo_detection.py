#### 예제 코드: 학습된 YOLOv8 모델을 사용한 객체 탐지
import cv2
from ultralytics import YOLO
# 학습된 YOLO 모델 로드
model = YOLO("runs/detect/train/weights/best.pt")
# 테스트할 이미지 불러오기
image_path = "week_3/test_image.jpg"
image = cv2.imread(image_path)
# 객체 탐지 실행
results = model(image)
# 탐지된 객체 시각화
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # 좌표값 변환
        label = result.names[int(box.cls[0])] # 클래스 라벨
        confidence = box.conf[0] # 신뢰도
        # 객체 경계 상자 그리기
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

# 결과 출력
cv2.imshow("YOLO Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()