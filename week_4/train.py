import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from autoencoder import ConvAutoencoder
import numpy as np

# 하이퍼파라미터
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
IMG_SIZE = 128
PATIENCE = 3   # 개선되지 않는 epoch 허용 횟수

# 데이터셋 (피카츄 정상 이미지만 학습)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
train_dataset = datasets.ImageFolder("week_4/dataset/train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 모델
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Early Stopping 변수
best_loss = np.inf
patience_counter = 0

# 학습
for epoch in range(EPOCHS):
    loss_sum = 0
    model.train()
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()

    avg_loss = loss_sum / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss={avg_loss:.4f}")

    # Early Stopping 체크
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "week_4/autoencoder.pth")  # 성능 향상 시 저장
        print(f"✅ 모델 개선됨 (Loss={best_loss:.4f}), 저장 완료")
    else:
        patience_counter += 1
        print(f"⏸️ 개선 없음 (patience {patience_counter}/{PATIENCE})")

    if patience_counter >= PATIENCE:
        print("⏹️ Early stopping 발동!")
        break

print("학습 종료")
