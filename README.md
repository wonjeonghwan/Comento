# Comento 

### Week_01 : Preprocessing
   - Hugging Face의 Food-101 데이터셋을 불러와 전처리를 수행합니다.  
   ### Basic
   - 이미지 내 붉은색 영역만을 검출
   ### Advanced
   - Food-101 데이터셋에서 임의의 5장을 선정하여 아래와 같은 이미지 처리 진행<br>
   사이즈 조정, Grayscale, 노이즈 감소(Denoise), 좌우 반전을 실행

### Week_02 : Unit test
   - 특정 이미지에 임의의 Depth_map값을 부여하고, Pytest를 활용하여 Unit test실행

### Week_03 : OpenCV Visualizing
   - Smoke-fire-detection dataset을 Yolov8n모델에 학습시키고<br>
   데이터 증강을 통해 성능 향상
   - Yolo_train_Aug.py : dataset 을 증강시켜 학습시킨 후, 원본을 추가로 학습진행
   - TP가 5~10% 가량 개선됨을 확인