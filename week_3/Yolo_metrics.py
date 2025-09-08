from ultralytics import YOLO
import matplotlib.pyplot as plt

def main():
    # 학습된 모델 불러오기
    model = YOLO("runs/detect/train/weights/best.pt")

    # 성능 평가
    metrics = model.val()
    print(metrics.results_dict)  # dict 형태 출력

    # dict로 변환
    results = metrics.results_dict

    # Precision, Recall 값 가져오기
    precision = results['metrics/precision(B)']
    recall = results['metrics/recall(B)']

    # 그래프 출력
    plt.plot([precision], label="Precision", marker='o')
    plt.plot([recall], label="Recall", marker='o')
    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Model Performance")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
