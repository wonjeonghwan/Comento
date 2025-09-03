# Comento Week_01 Preprocessing
Hugging Face의 Food-101 데이터셋을 불러와 전처리를 수행합니다.  

Basic : 기본 업무
Advanced : 심화 업무

## Basic

1. **week_01.py**  
   - 이미지 내 붉은색 영역만 추출.
   - 실행 시 이미지 내 붉은색만 처리하여 **원본이미지** 와 **처리 후 이미지**를 함꼐 띄워줌

## Advanced

1. **image_preprocessing.py**  
   - 기본 실행 프로그램.
   - Food - 101에서 무작위 이미지 5개를 가져와 사이즈 조정, Grayscale, 노이즈 감소(Denoise), 좌우 반전을 실행
   - 실행 시 이미지 처리 과정이 Preprocessed_output 폴더 내 각 과정별로 저장  