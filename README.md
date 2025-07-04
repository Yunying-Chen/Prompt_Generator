# Prompt_Generator
A modular pipeline for generating well-distributed prompts to perform SAM (Segment Anything Model) based segmentation, using sparse 3D-projected annotations.

## Requirements

SAM

## Files

- Inference.py: Generates segmentation masks from projected 3D points
- Generator.py: Creates sparse prompts based on point distribution or object polygons



## Input

1. Original Images: Shape [W, H, 3]
2. CSV files: .csv file with format [x, y, class_id]
3. Pre-trained model



## Performance

<table> <tr> <td><b>Original Image</b></td> <td><b>Annotations</b></td> </tr> <tr> <td><img src="./imgs/original_img.jpg" width="300"/></td> <td><img src="./imgs/annotation.jpg" width="300"/></td> </tr> <tr> <td><b>Generated Prompts</b></td> <td><b>Overlay with Predicted Mask</b></td> </tr> <tr> <td><img src="./imgs/prompts.jpg" width="300"/></td> <td><img src="./imgs/overlay_mask.jpg" width="300"/></td> </tr> <tr> <td colspan="2" align="center"><b>Final Mask</b></td> </tr> <tr> <td colspan="2" align="center"><img src="./imgs/mask.png" width="300"/></td> </tr> </table>
