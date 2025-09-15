import os, json, random
import numpy as np
import torch
from pathlib import Path

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def reconstruction_error(x, xrec):  # batch-wise MSE
    return ((x - xrec) ** 2).mean(dim=(1,2,3))

def save_threshold(errs, percentile, out_path):
    thr = float(np.percentile(errs, percentile))
    meta = {"percentile": percentile, "threshold": thr, "mean_err": float(np.mean(errs))}
    with open(out_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta

def load_threshold(path):
    with open(path) as f:
        return json.load(f)["threshold"]

def collect_images(path):
    IMG_EXT = {".jpg",".jpeg",".png",".bmp",".webp"}
    p = Path(path)
    if p.is_dir():
        return sorted([q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT])
    elif p.is_file():
        return [p]
    else:
        raise FileNotFoundError(f"{path} not found")

def device_auto():
    return "cuda" if torch.cuda.is_available() else "cpu"
