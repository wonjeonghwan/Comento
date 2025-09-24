import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from autoencoder import ConvAutoencoder
import numpy as np
import os

# 설정
IMG_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 (Normal + Anomaly 폴더 나눠서 넣어보기)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
test_dataset = datasets.ImageFolder("week_4/dataset/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 클래스 이름 매핑 (0=Normal, 1=Anomaly)
class_names = test_dataset.classes  # ["Normal", "Anomaly"]

# 모델 불러오기
model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load("week_4/autoencoder.pth", map_location=device))
model.eval()

# 손실 함수
criterion = torch.nn.MSELoss(reduction="mean")

# --- 1. 먼저 Normal 데이터의 재구성 오류 분포 수집 ---
normal_errors = []
all_results = []

for i, (imgs, labels) in enumerate(test_loader):
    imgs = imgs.to(device)
    outputs = model(imgs)
    loss = criterion(outputs, imgs).item()

    # 파일 경로에서 파일명 추출
    filepath, _ = test_dataset.samples[i]
    filename = os.path.basename(filepath)

    all_results.append((filename, labels.item(), loss))
    if labels.item() == 0:  # Normal
        normal_errors.append(loss)

if len(normal_errors) > 0:
    threshold = np.percentile(normal_errors, 90)
else:
    threshold = 0.1  # fallback 값
print(f"\n[INFO] Threshold set to {threshold:.6f} (95th percentile of Normal errors)\n")

# --- 3. 결과 출력 ---
for filename, label, loss in all_results:
    pred = 0 if loss <= threshold else 1   # 0=Normal, 1=Anomaly
    print(f"File={filename:20s} | True={class_names[label]:7s} | "
          f"Recon Error={loss:.6f} | Pred={class_names[pred]}")