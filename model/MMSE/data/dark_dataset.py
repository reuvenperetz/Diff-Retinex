import os
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


def _get_paths(path):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = []
    for root, _, fnames in os.walk(path):
        for fname in sorted(fnames):
            if fname.lower().endswith(exts):
                files.append(os.path.join(root, fname))
    if not files:
        raise RuntimeError(f"No images found in {path}")
    return sorted(files)


class MMSEDataset(Dataset):
    def __init__(self, dataroot, split="train", with_r=True, with_l=True, require_high=True):
        self.split = split
        self.with_r = with_r
        self.with_l = with_l

        base = os.path.join(dataroot, split)
        self.low_paths = _get_paths(os.path.join(base, "low"))
        self.r_paths = _get_paths(os.path.join(base, "R")) if with_r else None
        self.l_paths = _get_paths(os.path.join(base, "L")) if with_l else None
        self.high_paths = _get_paths(os.path.join(base, "high")) if require_high else None
        self.r_high_paths = _get_paths(os.path.join(base, "R_high")) if with_r and require_high else None
        self.l_high_paths = _get_paths(os.path.join(base, "L_high")) if with_l and require_high else None

        if with_r and len(self.low_paths) != len(self.r_paths):
            raise RuntimeError("low and R counts do not match")
        if with_l and len(self.low_paths) != len(self.l_paths):
            raise RuntimeError("low and L counts do not match")
        if require_high:
            if len(self.low_paths) != len(self.high_paths):
                raise RuntimeError("low and high counts do not match")
            if with_r and len(self.low_paths) != len(self.r_high_paths):
                raise RuntimeError("low and R_high counts do not match")
            if with_l and len(self.low_paths) != len(self.l_high_paths):
                raise RuntimeError("low and L_high counts do not match")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low = Image.open(self.low_paths[idx]).convert("RGB")
        low = self.to_tensor(low)

        sample = {"low": low}

        if self.with_r:
            r = Image.open(self.r_paths[idx]).convert("RGB")
            r = self.to_tensor(r)
            sample["R"] = r
            if self.r_high_paths:
                r_high = Image.open(self.r_high_paths[idx]).convert("RGB")
                r_high = self.to_tensor(r_high)
                sample["R_high"] = r_high

        if self.with_l:
            l = Image.open(self.l_paths[idx]).convert("L")
            l = self.to_tensor(l)
            sample["L"] = l
            if self.l_high_paths:
                l_high = Image.open(self.l_high_paths[idx]).convert("L")
                l_high = self.to_tensor(l_high)
                sample["L_high"] = l_high

        return sample
