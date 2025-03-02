import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from depth_anything_v2.dpt import DepthAnythingV2

# -------------------------------
# Helpers for red mask + YOLO
# -------------------------------
def create_red_mask(image_bgr):
    """
    Create a binary mask for red regions in the given BGR image.
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    return red_mask

def detect_objects_in_mask(image_bgr, mask, model):
    """
    Detect objects only within red mask bounding regions.
    Returns list of (x1, y1, x2, y2, conf, label).
    """
    detections = []
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 2 or h < 2:
            continue  # skip tiny areas
        
        roi = image_bgr[y:y+h, x:x+w]
        results = model(roi)
        
        # YOLO returns bounding boxes relative to the ROI
        for *xyxy, conf, cls_idx in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            label = model.names[int(cls_idx)]
            
            # Convert ROI coords back to full image coords
            global_x1 = x + x1
            global_y1 = y + y1
            global_x2 = x + x2
            global_y2 = y + y2
            
            detections.append((global_x1, global_y1, global_x2, global_y2, float(conf), label))
    
    return detections

def download_yolov5_if_needed():
    """
    Download YOLOv5 from GitHub if 'yolov5' directory is not found.
    """
    if not os.path.exists('yolov5'):
        print('Downloading YOLOv5...')
        os.system('git clone https://github.com/ultralytics/yolov5')
        os.system('pip install -r yolov5/requirements.txt')
    else:
        print('YOLOv5 directory already exists')

def load_yolo_model(conf_threshold=0.25, iou_threshold=0.45):
    """
    Load YOLOv5 (local if cloned, or directly from ultralytics if needed).
    """
    model = torch.hub.load('yolov5', 'yolov5s', source='local')
    model.conf = conf_threshold
    model.iou = iou_threshold
    return model

# -------------------------------
# Main Depth + YOLO pipeline
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 + YOLOv5 + Red Highlight Detection')
    parser.add_argument('--img-path', type=str, required=True, help='Path to input image or directory')
    parser.add_argument('--input-size', type=int, default=518, help='Input size for Depth Anything model')
    parser.add_argument('--outdir', type=str, default='./vis_depth', help='Output directory')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help='Encoder model size')
    parser.add_argument('--pred-only', action='store_true', help='Save only depth map (no side-by-side)')
    parser.add_argument('--grayscale', action='store_true', help='Generate grayscale depth map (no color)')
    parser.add_argument('--detect-red', action='store_true', help='Run YOLO detection in red-highlighted areas')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {DEVICE}')
    
    # 1. Download & load YOLOv5
    download_yolov5_if_needed()
    yolo_model = load_yolo_model()
    
    # 2. Load DepthAnythingV2
    model_configs = {
        # 'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        # 'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [152, 304, 608, 1216]},
        # 'vitg': {'encoder': 'vitg', 'features': 256, 'out_channels': [192, 384, 768, 1536]}
    }
    
    depth_model = DepthAnythingV2(**model_configs[args.encoder])
    ckpt_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    depth_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    depth_model = depth_model.to(DEVICE).eval()
    
    # 3. Gather input images
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('.txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        # If directory, gather all image files
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        filenames = [f for f in filenames if os.path.isfile(f) and f.lower().endswith(valid_exts)]
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 4. For color mapping the depth
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # 5. Process each image
    for idx, fname in enumerate(filenames):
        print(f'[{idx+1}/{len(filenames)}] Processing: {fname}')
        raw_image = cv2.imread(fname)
        if raw_image is None:
            print(f'Could not read image: {fname}')
            continue
        
        # 5a. Generate depth
        depth_map = depth_model.infer_image(raw_image, args.input_size)
        depth_map = depth_map - depth_map.min()
        depth_map = depth_map / (depth_map.max() + 1e-8) * 255.0
        depth_map = depth_map.astype(np.uint8)
        
        # 5b. Convert depth map to color or grayscale
        if args.grayscale:
            depth_vis = np.repeat(depth_map[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        # 5c. If requested, detect objects in red-highlighted regions of depth_vis
        #     (assuming the user has drawn red on the depth map to highlight an object)
        if args.detect_red:
            red_mask = create_red_mask(depth_vis)
            detections = detect_objects_in_mask(depth_vis, red_mask, yolo_model)
            
            # Draw bounding boxes on the depth map
            for (x1, y1, x2, y2, conf, label) in detections:
                cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(depth_vis, f'{label} {conf:.2f}', 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
        
        # 5d. Save outputs
        out_name = os.path.splitext(os.path.basename(fname))[0] + '_depth.png'
        out_path = os.path.join(args.outdir, out_name)
        
        if args.pred_only:
            # Save only the depth map with bounding boxes if any
            cv2.imwrite(out_path, depth_vis)
        else:
            # Combine original + depth side by side
            # Resize original to match depth height
            if raw_image.shape[0] != depth_vis.shape[0]:
                scale = depth_vis.shape[0] / raw_image.shape[0]
                new_w = int(raw_image.shape[1] * scale)
                resized_raw = cv2.resize(raw_image, (new_w, depth_vis.shape[0]))
            else:
                resized_raw = raw_image
            
            # Create a small white gap
            gap = np.ones((depth_vis.shape[0], 50, 3), dtype=np.uint8) * 255
            combined = cv2.hconcat([resized_raw, gap, depth_vis])
            
            cv2.imwrite(out_path, combined)
        
        print(f'Saved output to: {out_path}')

if __name__ == '__main__':
    main()
