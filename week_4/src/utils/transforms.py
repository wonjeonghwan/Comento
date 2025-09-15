from torchvision import transforms

def build_transform(img_size=128, aug=False):
    t = []
    if aug:
        t += [
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
        ]
    else:
        t += [transforms.Resize((img_size, img_size))]
    t += [
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
    ]
    return transforms.Compose(t)
