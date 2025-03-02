import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2

def create_red_mask(image_bgr):
    """
    Create a binary mask for red regions in the given BGR image.
    Adjusted to capture reddish–orange tones.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 80, 50])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 80, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    return red_mask

def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2: Red Region Distance Estimation (in cm)")
    parser.add_argument('--img-path', type=str, required=True, help="Path to input image or directory")
    parser.add_argument('--input-size', type=int, default=518, help="Input size for Depth Anything model")
    parser.add_argument('--outdir', type=str, default='./vis_depth', help="Output directory")
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help="Encoder model size")
    parser.add_argument('--pred-only', action='store_true', help="Save only depth map (no side-by-side)")
    parser.add_argument('--grayscale', action='store_true', help="Generate grayscale depth map (no color)")
    parser.add_argument('--detect-red', action='store_true', help="Process red region for distance estimation")
    # Use a single calibration factor (raw_depth * factor gives distance in meters)
    parser.add_argument('--calib-factor', type=float, default=0.03,
                        help="Multiplicative factor to convert raw depth to meters (default ~0.03)")
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Load DepthAnythingV2 model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [152, 304, 608, 1216]},
        'vitg': {'encoder': 'vitg', 'features': 256, 'out_channels': [192, 384, 768, 1536]}
    }
    depth_model = DepthAnythingV2(**model_configs[args.encoder])
    ckpt_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    depth_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()

    # Gather input image(s)
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('.txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        filenames = [f for f in filenames if os.path.isfile(f) and f.lower().endswith(valid_exts)]

    os.makedirs(args.outdir, exist_ok=True)
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for idx, fname in enumerate(filenames):
        print(f"[{idx+1}/{len(filenames)}] Processing: {fname}")
        raw_image = cv2.imread(fname)
        if raw_image is None:
            print(f"Could not read image: {fname}")
            continue

        # Obtain raw depth map (raw_depth is in arbitrary units)
        raw_depth = depth_model.infer_image(raw_image, args.input_size)
        # Normalize depth for visualization (0-255)
        norm_depth = raw_depth - raw_depth.min()
        norm_depth = norm_depth / (norm_depth.max() + 1e-8) * 255.0
        norm_depth = norm_depth.astype(np.uint8)

        if args.grayscale:
            depth_vis = np.repeat(norm_depth[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(norm_depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        if args.detect_red:
            red_mask = create_red_mask(depth_vis)
            debug_mask_path = os.path.join(args.outdir, f"debug_red_mask_{idx}.png")
            cv2.imwrite(debug_mask_path, red_mask)
            print(f"Debug red mask saved to: {debug_mask_path}")

            indices = np.where(red_mask > 0)
            if indices[0].size == 0:
                print("No red region detected.")
            else:
                # Compute median raw depth in red region
                median_raw_depth = np.median(raw_depth[indices])
                # Convert raw depth to meters using calibration factor
                estimated_distance_m = median_raw_depth * args.calib_factor
                # Convert to centimeters
                distance_cm = estimated_distance_m * 100.0
                print(f"Estimated distance to red region: {distance_cm:.2f} cm")
                cv2.putText(depth_vis, f"Distance: {distance_cm:.2f} cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                overlay = depth_vis.copy()
                overlay[indices] = [0, 0, 255]  # red overlay
                depth_vis = cv2.addWeighted(depth_vis, 0.7, overlay, 0.3, 0)

        # Save the output image
        out_name = os.path.splitext(os.path.basename(fname))[0] + "_depth.png"
        out_path = os.path.join(args.outdir, out_name)
        if args.pred_only:
            cv2.imwrite(out_path, depth_vis)
        else:
            if raw_image.shape[0] != depth_vis.shape[0]:
                scale = depth_vis.shape[0] / raw_image.shape[0]
                new_w = int(raw_image.shape[1] * scale)
                resized_raw = cv2.resize(raw_image, (new_w, depth_vis.shape[0]))
            else:
                resized_raw = raw_image
            gap = np.ones((depth_vis.shape[0], 50, 3), dtype=np.uint8) * 255
            combined = cv2.hconcat([resized_raw, gap, depth_vis])
            cv2.imwrite(out_path, combined)
        print(f"Saved output to: {out_path}")

if __name__ == "__main__":
    main()
