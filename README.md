# Depth-Anything-V2

## Pre-trained Models

Create folder : checkpoints and copy the file in that folder:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |

## Command to run the file:

```bash
python main.py --encoder vitl --img-path assets/examples --outdir depth_vis
```
