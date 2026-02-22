import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from networks import build_mmse_net
from data.dark_dataset import MMSEDataset


def select_device(device_pref):
    pref = device_pref.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, device, train=True):
    total = 0.0
    count = 0
    if train:
        model.train()
    else:
        model.eval()
    for batch in loader:
        if "R" in batch:
            inp = batch["R"].to(device) * 2.0 - 1.0
            target = batch["R_high"].to(device) * 2.0 - 1.0
        elif "L" in batch:
            inp = batch["L"].to(device) * 2.0 - 1.0
            target = batch["L_high"].to(device) * 2.0 - 1.0
        else:
            raise RuntimeError("batch missing R/L targets")
        if train:
            model.optimizer.zero_grad()
        pred = model(inp)
        loss = F.l1_loss(pred, target)
        if train:
            loss.backward()
            model.optimizer.step()
        total += loss.item()
        count += 1
    return total / max(1, count)


def train_one(model, train_loader, val_loader, device, epochs, save_path):
    best = 1e9
    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, device, train=True)
        val_loss = run_epoch(model, val_loader, device, train=False) if val_loader else train_loss
        if val_loss < best:
            best = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        print(f"epoch {epoch + 1}: train {train_loss:.6f} val {val_loss:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="dataset root with train/val subfolders")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "cnn"])
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda", help="cuda, mps, or cpu")
    parser.add_argument("--train-r", action="store_true", help="train reflectance MMSE")
    parser.add_argument("--train-l", action="store_true", help="train illumination MMSE")
    parser.add_argument("--r-weights", type=str, default="weights/mmse_r.pth")
    parser.add_argument("--l-weights", type=str, default="weights/mmse_l.pth")

    args = parser.parse_args()
    if not args.train_r and not args.train_l:
        raise SystemExit("select --train-r and/or --train-l")

    device = select_device(args.device)

    if args.train_r:
        train_set = MMSEDataset(args.data_root, split="train", with_r=True, with_l=False, require_high=True)
        val_set = MMSEDataset(args.data_root, split="val", with_r=True, with_l=False, require_high=True) if os.path.isdir(os.path.join(args.data_root, "val")) else None
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) if val_set else None

        model_r = build_mmse_net(args.arch, in_ch=3, out_ch=3, base_ch=args.base_ch)
        model_r.optimizer = torch.optim.Adam(model_r.parameters(), lr=args.lr)
        model_r.to(device)
        train_one(model_r, train_loader, val_loader, device, args.epochs, args.r_weights)

    if args.train_l:
        train_set = MMSEDataset(args.data_root, split="train", with_r=False, with_l=True, require_high=True)
        val_set = MMSEDataset(args.data_root, split="val", with_r=False, with_l=True, require_high=True) if os.path.isdir(os.path.join(args.data_root, "val")) else None
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) if val_set else None

        model_l = build_mmse_net(args.arch, in_ch=1, out_ch=1, base_ch=args.base_ch)
        model_l.optimizer = torch.optim.Adam(model_l.parameters(), lr=args.lr)
        model_l.to(device)
        train_one(model_l, train_loader, val_loader, device, args.epochs, args.l_weights)


if __name__ == "__main__":
    main()
