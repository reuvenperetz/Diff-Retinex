import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(cmd, cwd):
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmse-data-root", type=str, required=True, help="MMSE dataset root")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "cnn"])
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recon-weight", type=float, default=0.0)
    parser.add_argument("--train-r", action="store_true", help="train reflectance MMSE")
    parser.add_argument("--train-l", action="store_true", help="train illumination MMSE")
    parser.add_argument("--chain-config", type=str, default="config/Diff_Retinex_val.json",
                        help="config for full-chain validation")
    parser.add_argument("--mmse-components", type=str, default="both", choices=["r", "l", "both"])
    parser.add_argument("--mmse-arg", action="append", default=[],
                        help="extra args passed to train_mmse.py (repeatable, use --mmse-arg=--flag)")
    parser.add_argument("--mmse-args", nargs=argparse.REMAINDER, default=[],
                        help="extra args passed to train_mmse.py after --mmse-args")
    parser.add_argument("--chain-args", nargs=argparse.REMAINDER, default=[],
                        help="extra args passed to test_from_dataset.py after --chain-args")
    args = parser.parse_args()

    if not args.train_r and not args.train_l:
        args.train_r = True
        args.train_l = True

    repo_root = Path(__file__).resolve().parents[2]
    run_root = repo_root / "experiments" / f"mmse_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    weights_dir = run_root / "weights"
    vis_dir = run_root / "mmse_vis"
    weights_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    r_weights = weights_dir / "mmse_r_best_psnr.pth"
    l_weights = weights_dir / "mmse_l_best_psnr.pth"

    train_cmd = [
        sys.executable, "model/MMSE/train_mmse.py",
        "--data-root", args.mmse_data_root,
        "--epochs", str(args.epochs),
        "--arch", args.arch,
        "--base-ch", str(args.base_ch),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", args.device,
        "--recon-weight", str(args.recon_weight),
        "--val-vis-every", "1",
        "--val-vis-indices", "0,1",
        "--val-vis-dir", str(vis_dir),
        "--r-weights", str(r_weights),
        "--l-weights", str(l_weights),
    ]
    if args.train_r:
        train_cmd.append("--train-r")
    if args.train_l:
        train_cmd.append("--train-l")

    if args.mmse_arg:
        train_cmd.extend(args.mmse_arg)
    if args.mmse_args:
        train_cmd.extend(args.mmse_args)

    run(train_cmd, cwd=repo_root)

    mmse_components = args.mmse_components
    if mmse_components == "both" and not (args.train_r and args.train_l):
        mmse_components = "r" if args.train_r else "l"

    chain_cmd = [
        sys.executable, "test_from_dataset.py",
        "--config", args.chain_config,
        "--use_mmse",
        "--mmse_components", mmse_components,
        "--mmse_arch", args.arch,
        "--mmse_r_weights", str(r_weights),
        "--mmse_l_weights", str(l_weights),
        "--mmse_base_ch", str(args.base_ch),
    ]
    if args.chain_args:
        chain_cmd.extend(args.chain_args)
    run(chain_cmd, cwd=repo_root)


if __name__ == "__main__":
    main()
