from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from utils.transforms import build_transform  # 기존 함수 그대로 유지

class ImageFolderNoLabel(Dataset):
    IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir, img_size=128, aug=False, tf=None):
        """
        tf: 외부에서 넘겨주는 torchvision transforms.Compose
            - 있으면 tf 우선 사용
            - 없으면 기존 build_transform(img_size, aug) 사용
        """
        self.root = Path(root_dir)
        self.paths = sorted([p for p in self.root.rglob("*")
                             if p.suffix.lower() in self.IMG_EXT])
        if not self.paths:
            raise FileNotFoundError(f"No images in {self.root}")

        # tf를 주입받으면 그걸 쓰고, 아니면 기존 build_transform 사용
        self.tf = tf if tf is not None else build_transform(img_size, aug=aug)

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _rgba_to_rgb_white(img: Image.Image) -> Image.Image:
        # PNG 투명 배경을 흰색으로 일관 변환 (학습/추론 모두 동일 처리)
        if img.mode == "RGBA":
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(bg, img)
        return img.convert("RGB")

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGBA")  # 알파 고려
        img = self._rgba_to_rgb_white(img)
        return self.tf(img)
