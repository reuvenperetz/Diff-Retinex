import argparse
import os
import torch
import torch.nn.functional as F
import warnings
import model.Diff_RDA.core.metrics as Metrics
from PIL import Image
from torch.utils.data import DataLoader
from networks import build_mmse_net
from data.dark_dataset import MMSEDataset
from datetime import datetime
from contextlib import contextmanager


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


@contextmanager
def mlflow_run(enabled, experiment_name, run_name, tags=None):
    if not enabled:
        yield None
        return
    try:
        import mlflow
    except Exception as e:
        print(f"MLflow disabled: {e}")
        yield None
        return
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        if tags:
            mlflow.set_tags(tags)
        yield mlflow


def run_epoch(model, loader, device, component, recon_weight=0.0, train=True):
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
        loss = F.mse_loss(pred, target)
        if recon_weight > 0.0:
            if component == "r":
                if "L_high" not in batch or "high" not in batch:
                    raise SystemExit("Missing L_high/high in batch for reconstruction loss.")
                l_high = batch["L_high"].to(device).clamp(0.0, 1.0)
                r_pred = (pred.clamp(-1.0, 1.0) + 1.0) * 0.5
                recon = (r_pred * l_high).clamp(0.0, 1.0)
            else:
                if "R_high" not in batch or "high" not in batch:
                    raise SystemExit("Missing R_high/high in batch for reconstruction loss.")
                r_high = batch["R_high"].to(device).clamp(0.0, 1.0)
                l_pred = (pred.clamp(-1.0, 1.0) + 1.0) * 0.5
                recon = (r_high * l_pred).clamp(0.0, 1.0)
            high = batch["high"].to(device).clamp(0.0, 1.0)
            recon_loss = F.mse_loss(recon, high)
            loss = loss + recon_weight * recon_loss
        if train:
            loss.backward()
            model.optimizer.step()
        total += loss.item()
        count += 1
    return total / max(1, count)

def eval_metrics(model, loader, device, component, lpips_model=None, skimage_niqe=None):
    model.eval()
    recon_psnr_sum = 0.0
    recon_ssim_sum = 0.0
    comp_psnr_sum = 0.0
    comp_ssim_sum = 0.0
    lpips_sum = 0.0
    niqe_sum = 0.0
    loss_sum = 0.0
    count = 0

    for batch in loader:
        if component == "r":
            inp = batch["R"].to(device) * 2.0 - 1.0
            target = batch["R_high"].to(device) * 2.0 - 1.0
        else:
            inp = batch["L"].to(device) * 2.0 - 1.0
            target = batch["L_high"].to(device) * 2.0 - 1.0
        with torch.no_grad():
            pred = model(inp).clamp(-1.0, 1.0)
            loss_sum += F.l1_loss(pred, target).item()

        pred_img = ((pred.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()
        gt_comp_img = ((target.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()

        if component == "r":
            if "L_high" not in batch:
                raise SystemExit("Missing L_high in validation batch. Rebuild MMSE dataset with --include-high.")
            l_high = batch["L_high"]
            l_high = l_high.to(device).clamp(0.0, 1.0)
            recon = (pred.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5
            recon = (recon * l_high.squeeze(0)).clamp(0.0, 1.0).cpu()
        else:
            if "R_high" not in batch:
                raise SystemExit("Missing R_high in validation batch. Rebuild MMSE dataset with --include-high.")
            r_high = batch["R_high"].to(device).clamp(0.0, 1.0)
            l_pred = (pred.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5
            recon = (r_high.squeeze(0) * l_pred).clamp(0.0, 1.0).cpu()
        gt_img = batch["high"].squeeze(0).clamp(0.0, 1.0)

        if pred_img.shape[0] == 1:
            pred_np = (pred_img.squeeze(0).numpy() * 255.0).astype("uint8")
            gt_comp_np = (gt_comp_img.squeeze(0).numpy() * 255.0).astype("uint8")
        else:
            pred_np = (pred_img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            gt_comp_np = (gt_comp_img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")

        comp_psnr_sum += Metrics.calculate_psnr(pred_np, gt_comp_np)
        comp_ssim_sum += Metrics.calculate_ssim(pred_np, gt_comp_np)

        recon_np = (recon.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        gt_np = (gt_img.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        recon_psnr_sum += Metrics.calculate_psnr(recon_np, gt_np)
        recon_ssim_sum += Metrics.calculate_ssim(recon_np, gt_np)

        if lpips_model is not None:
            with torch.no_grad():
                pred_lp = recon
                gt_lp = gt_img
                pred_lp = pred_lp.unsqueeze(0) * 2.0 - 1.0
                gt_lp = gt_lp.unsqueeze(0) * 2.0 - 1.0
                lpips_sum += lpips_model(pred_lp.to(device), gt_lp.to(device)).item()

        if skimage_niqe is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gray = (0.299 * recon[0] + 0.587 * recon[1] + 0.114 * recon[2]).numpy()
                niqe_sum += float(skimage_niqe(gray))

        count += 1

    if count == 0:
        return None
    return {
        "val_loss": loss_sum / count,
        "comp_psnr": comp_psnr_sum / count,
        "comp_ssim": comp_ssim_sum / count,
        "recon_psnr": recon_psnr_sum / count,
        "recon_ssim": recon_ssim_sum / count,
        "lpips": lpips_sum / count if lpips_model is not None else None,
        "niqe": niqe_sum / count if skimage_niqe is not None else None,
    }

def save_vis(model, device, val_dataset, indices, out_dir, epoch):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    for idx in indices:
        if idx < 0 or idx >= len(val_dataset):
            continue
        sample = val_dataset[idx]
        if "R" in sample:
            inp = sample["R"].unsqueeze(0).to(device) * 2.0 - 1.0
            gt = sample["R_high"]
        else:
            inp = sample["L"].unsqueeze(0).to(device) * 2.0 - 1.0
            gt = sample["L_high"]
        with torch.no_grad():
            pred = model(inp).clamp(-1.0, 1.0)
        inp_img = ((inp.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()
        pred_img = ((pred.squeeze(0).clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()
        gt_img = gt.clamp(0.0, 1.0)

        def to_pil(t):
            if t.shape[0] == 1:
                arr = (t.squeeze(0).numpy() * 255.0).astype("uint8")
                return Image.fromarray(arr, mode="L")
            arr = (t.permute(1, 2, 0).numpy() * 255.0).astype("uint8")
            return Image.fromarray(arr, mode="RGB")

        inp_pil = to_pil(inp_img)
        pred_pil = to_pil(pred_img)
        gt_pil = to_pil(gt_img)

        w = inp_pil.width + pred_pil.width + gt_pil.width
        h = max(inp_pil.height, pred_pil.height, gt_pil.height)
        mode = "L" if inp_pil.mode == "L" else "RGB"
        canvas = Image.new(mode, (w, h))
        x = 0
        canvas.paste(inp_pil, (x, 0))
        x += inp_pil.width
        canvas.paste(pred_pil, (x, 0))
        x += pred_pil.width
        canvas.paste(gt_pil, (x, 0))

        base = os.path.join(out_dir, f"epoch_{epoch:04d}_idx_{idx}")
        canvas.save(base + "_compare.png")


def train_one(model, train_loader, val_loader, device, epochs, save_path,
              component, recon_weight=0.0,
              val_dataset=None, vis_every=0, vis_indices=None, vis_dir=None,
              mlflow_client=None):
    best_loss = 1e9
    best_recon_psnr = -1.0
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').to(device)
        lpips_model.eval()
    except Exception:
        lpips_model = None

    try:
        from skimage.metrics import niqe as skimage_niqe
    except Exception:
        skimage_niqe = None

    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, device, component, recon_weight, train=True)
        metrics = eval_metrics(model, val_loader, device, component, lpips_model, skimage_niqe) if val_loader else None
        val_loss = metrics["val_loss"] if metrics else train_loss
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
        if vis_every > 0 and val_dataset is not None and vis_indices:
            if (epoch + 1) % vis_every == 0:
                save_vis(model, device, val_dataset, vis_indices, vis_dir, epoch + 1)
                if mlflow_client is not None:
                    try:
                        mlflow_client.log_artifacts(vis_dir, artifact_path="val_vis")
                    except Exception:
                        pass
        if metrics:
            msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Comp PSNR: {metrics['comp_psnr']:.2f}, Comp SSIM: {metrics['comp_ssim']:.4f}, Recon PSNR: {metrics['recon_psnr']:.2f}, Recon SSIM: {metrics['recon_ssim']:.4f}"
            if metrics['lpips'] is not None:
                msg += f", LPIPS: {metrics['lpips']:.4f}"
            if metrics['niqe'] is not None:
                msg += f", NIQE: {metrics['niqe']:.4f}"
            print(msg)
            if mlflow_client is not None:
                mlflow_client.log_metric("train_loss", train_loss, step=epoch + 1)
                mlflow_client.log_metric("val_loss", val_loss, step=epoch + 1)
                mlflow_client.log_metric("comp_psnr", metrics["comp_psnr"], step=epoch + 1)
                mlflow_client.log_metric("comp_ssim", metrics["comp_ssim"], step=epoch + 1)
                mlflow_client.log_metric("recon_psnr", metrics["recon_psnr"], step=epoch + 1)
                mlflow_client.log_metric("recon_ssim", metrics["recon_ssim"], step=epoch + 1)
                if metrics["lpips"] is not None:
                    mlflow_client.log_metric("lpips", metrics["lpips"], step=epoch + 1)
                if metrics["niqe"] is not None:
                    mlflow_client.log_metric("niqe", metrics["niqe"], step=epoch + 1)
            if metrics['recon_psnr'] > best_recon_psnr:
                best_recon_psnr = metrics['recon_psnr']
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)
                if mlflow_client is not None:
                    try:
                        mlflow_client.log_artifact(save_path, artifact_path="checkpoints")
                    except Exception:
                        pass
        else:
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if mlflow_client is not None:
                mlflow_client.log_metric("train_loss", train_loss, step=epoch + 1)
                mlflow_client.log_metric("val_loss", val_loss, step=epoch + 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, help="dataset root with train/val subfolders")
    parser.add_argument("--arch", type=str, default="unet", choices=["unet", "cnn"])
    parser.add_argument("--base-ch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--recon-weight", type=float, default=0.0,
                        help="weight for reconstruction MSE loss")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, mps, or cpu")
    parser.add_argument("--train-r", action="store_true", help="train reflectance MMSE")
    parser.add_argument("--train-l", action="store_true", help="train illumination MMSE")
    parser.add_argument("--r-weights", type=str, default="weights/mmse_r.pth")
    parser.add_argument("--l-weights", type=str, default="weights/mmse_l.pth")
    parser.add_argument("--val-vis-every", type=int, default=1,
                        help="save val visualizations every N epochs (0 disables)")
    parser.add_argument("--val-vis-indices", type=str, default="0,1",
                        help="comma-separated val indices to visualize")
    parser.add_argument("--val-vis-dir", type=str, default="mmse_vis",
                        help="output directory for visualization images")
    parser.add_argument("--mlflow", action="store_true", help="enable MLflow logging")
    parser.add_argument("--mlflow-exp", type=str, default="MMSE-Diffretinex",
                        help="MLflow experiment name")
    parser.add_argument("--mlflow-run", type=str, default="",
                        help="MLflow run name (default: auto)")

    # MLFLOW_TRACKING_URI=/path/to/mlruns python model/MMSE/train_mmse.py --data-root ...

    args = parser.parse_args()
    if not args.train_r and not args.train_l:
        raise SystemExit("select --train-r and/or --train-l")

    device = select_device(args.device)
    exp_root = os.path.join("experiments", f"mmse_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(exp_root, exist_ok=True)
    vis_dir = args.val_vis_dir
    if not os.path.isabs(vis_dir):
        vis_dir = os.path.join(exp_root, vis_dir)
    r_weights = args.r_weights
    l_weights = args.l_weights
    if not os.path.isabs(r_weights):
        r_weights = os.path.join(exp_root, r_weights)
    if not os.path.isabs(l_weights):
        l_weights = os.path.join(exp_root, l_weights)

    run_name = args.mlflow_run if args.mlflow_run else f"mmse_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow_run(args.mlflow, args.mlflow_exp, run_name, tags={"component": "both" if args.train_r and args.train_l else ("r" if args.train_r else "l")} ) as mlflow_client:
        if mlflow_client is not None:
            mlflow_client.log_params({
                "arch": args.arch,
                "base_ch": args.base_ch,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "recon_weight": args.recon_weight,
                "device": str(device),
                "data_root": args.data_root,
                "val_vis_every": args.val_vis_every,
                "val_vis_indices": args.val_vis_indices,
            })

        if args.train_r:
            train_set = MMSEDataset(args.data_root, split="train", with_r=True, with_l=False, require_high=True,
                                    with_r_high=True, with_l_high=False)
            val_set = MMSEDataset(args.data_root, split="val", with_r=True, with_l=False, require_high=True,
                                  with_r_high=True, with_l_high=True) if os.path.isdir(os.path.join(args.data_root, "val")) else None
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) if val_set else None

            model_r = build_mmse_net(args.arch, in_ch=3, out_ch=3, base_ch=args.base_ch)
            model_r.optimizer = torch.optim.Adam(model_r.parameters(), lr=args.lr)
            model_r.to(device)
            vis_indices = [int(x) for x in args.val_vis_indices.split(",") if x.strip() != ""]
            train_one(model_r, train_loader, val_loader, device, args.epochs, r_weights,
                      component="r", recon_weight=args.recon_weight,
                      val_dataset=val_set, vis_every=args.val_vis_every,
                      vis_indices=vis_indices, vis_dir=vis_dir,
                      mlflow_client=mlflow_client)

        if args.train_l:
            train_set = MMSEDataset(args.data_root, split="train", with_r=False, with_l=True, require_high=True,
                                    with_r_high=False, with_l_high=True)
            val_set = MMSEDataset(args.data_root, split="val", with_r=False, with_l=True, require_high=True,
                                  with_r_high=True, with_l_high=True) if os.path.isdir(os.path.join(args.data_root, "val")) else None
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) if val_set else None

            model_l = build_mmse_net(args.arch, in_ch=1, out_ch=1, base_ch=args.base_ch)
            model_l.optimizer = torch.optim.Adam(model_l.parameters(), lr=args.lr)
            model_l.to(device)
            vis_indices = [int(x) for x in args.val_vis_indices.split(",") if x.strip() != ""]
            train_one(model_l, train_loader, val_loader, device, args.epochs, l_weights,
                      component="l", recon_weight=args.recon_weight,
                      val_dataset=val_set, vis_every=args.val_vis_every,
                      vis_indices=vis_indices, vis_dir=vis_dir,
                      mlflow_client=mlflow_client)


if __name__ == "__main__":
    main()
