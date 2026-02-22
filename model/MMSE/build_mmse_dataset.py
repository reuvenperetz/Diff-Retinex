import argparse
import os
from PIL import Image
import torch
from torchvision import transforms


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


def iter_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    for root, _, files in os.walk(folder):
        for name in sorted(files):
            if name.lower().endswith(exts):
                yield os.path.join(root, name)


def save_tensor_rgb(t, path):
    t = t.clamp(0.0, 1.0)
    img = (t.permute(1, 2, 0).cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(img, mode="RGB").save(path)


def save_tensor_l(t, path):
    t = t.clamp(0.0, 1.0)
    img = (t.squeeze(0).cpu().numpy() * 255.0).astype("uint8")
    Image.fromarray(img, mode="L").save(path)


def process_split(model, device, input_low, output_root, split_name, max_images=None,
                  input_high=None, include_high=False):
    low_out = os.path.join(output_root, split_name, "low")
    r_out = os.path.join(output_root, split_name, "R")
    l_out = os.path.join(output_root, split_name, "L")
    os.makedirs(low_out, exist_ok=True)
    os.makedirs(r_out, exist_ok=True)
    os.makedirs(l_out, exist_ok=True)
    if include_high and input_high:
        high_out = os.path.join(output_root, split_name, "high")
        r_high_out = os.path.join(output_root, split_name, "R_high")
        l_high_out = os.path.join(output_root, split_name, "L_high")
        os.makedirs(high_out, exist_ok=True)
        os.makedirs(r_high_out, exist_ok=True)
        os.makedirs(l_high_out, exist_ok=True)
    else:
        high_out = r_high_out = l_high_out = None

    to_tensor = transforms.ToTensor()
    count = 0
    for path in iter_images(input_low):
        name = os.path.splitext(os.path.basename(path))[0]
        img = Image.open(path).convert("RGB")
        low = to_tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            r, l = model(low)

        save_tensor_rgb(r.squeeze(0), os.path.join(r_out, f"{name}.png"))
        save_tensor_l(l.squeeze(0), os.path.join(l_out, f"{name}.png"))
        img.save(os.path.join(low_out, f"{name}.png"))

        count += 1
        if max_images and count >= max_images:
            break

    if include_high and input_high:
        count = 0
        for path in iter_images(input_high):
            name = os.path.splitext(os.path.basename(path))[0]
            img = Image.open(path).convert("RGB")
            high = to_tensor(img).unsqueeze(0).to(device)

            with torch.no_grad():
                r, l = model(high)

            save_tensor_rgb(r.squeeze(0), os.path.join(r_high_out, f"{name}.png"))
            save_tensor_l(l.squeeze(0), os.path.join(l_high_out, f"{name}.png"))
            img.save(os.path.join(high_out, f"{name}.png"))

            count += 1
            if max_images and count >= max_images:
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, required=True,
                        help="dataset root with train/low and eval/low (or val/low)")
    parser.add_argument("--output-root", type=str, required=True,
                        help="output root for mmse_dataset")
    parser.add_argument("--tdn-weights", type=str, default="model/Diff_TDN/weights/checkpoint_LOL_Diff_TDN.pth")
    parser.add_argument("--device", type=str, default="cuda", help="cuda, mps, or cpu")
    parser.add_argument("--max-images", type=int, default=0, help="limit images per split (0 = all)")
    parser.add_argument("--use-eval-as-val", action="store_true",
                        help="use input eval/low as output val/low")
    parser.add_argument("--include-high", action="store_true", default=True,
                        help="also decompose high images into R_high/L_high")
    args = parser.parse_args()

    device = select_device(args.device)

    from model.Diff_TDN.TDN_network import DecomNet as create_model
    model = create_model().to(device)
    weights = torch.load(args.tdn_weights, map_location=device)
    state = weights["model"] if isinstance(weights, dict) and "model" in weights else weights
    model.load_state_dict(state)
    model.eval()

    train_low = os.path.join(args.input_root, "train", "low")
    train_high = os.path.join(args.input_root, "train", "high")
    val_low = os.path.join(args.input_root, "val", "low")
    val_high = os.path.join(args.input_root, "val", "high")
    eval_low = os.path.join(args.input_root, "eval", "low")
    eval_high = os.path.join(args.input_root, "eval", "high")

    if os.path.isdir(train_low):
        process_split(model, device, train_low, args.output_root, "train",
                      max_images=args.max_images or None,
                      input_high=train_high if os.path.isdir(train_high) else None,
                      include_high=args.include_high)
    else:
        raise SystemExit(f"missing {train_low}")

    if args.use_eval_as_val and os.path.isdir(eval_low):
        process_split(model, device, eval_low, args.output_root, "val",
                      max_images=args.max_images or None,
                      input_high=eval_high if os.path.isdir(eval_high) else None,
                      include_high=args.include_high)
    elif os.path.isdir(val_low):
        process_split(model, device, val_low, args.output_root, "val",
                      max_images=args.max_images or None,
                      input_high=val_high if os.path.isdir(val_high) else None,
                      include_high=args.include_high)
    elif os.path.isdir(eval_low):
        process_split(model, device, eval_low, args.output_root, "val",
                      max_images=args.max_images or None,
                      input_high=eval_high if os.path.isdir(eval_high) else None,
                      include_high=args.include_high)
    else:
        print("No val/eval split found, skipping val.")


if __name__ == "__main__":
    main()
