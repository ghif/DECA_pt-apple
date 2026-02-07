import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

def compare_images(img1_path, img2_path):
    if not os.path.exists(img1_path):
        print(f"Error: {img1_path} does not exist")
        return
    if not os.path.exists(img2_path):
        print(f"Error: {img2_path} does not exist")
        return

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1.shape != img2.shape:
        print(f"Shape mismatch: {img1.shape} vs {img2.shape}")
        # Resize img2 to match img1 for comparison if they are close
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    mse = np.mean((img1 - img2) ** 2)
    s = ssim(img1, img2, channel_axis=2)

    print(f"Comparison between {img1_path} and {img2_path}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  SSIM: {s:.4f}")

    # Break down by panels (assuming 6 panels horizontally)
    w = img1.shape[1]
    panel_w = w // 6
    for i in range(6):
        p1 = img1[:, i*panel_w:(i+1)*panel_w]
        p2 = img2[:, i*panel_w:(i+1)*panel_w]
        p_mse = np.mean((p1 - p2) ** 2)
        p_ssim = ssim(p1, p2, channel_axis=2)
        print(f"  Panel {i+1} MSE: {p_mse:.4f}, SSIM: {p_ssim:.4f}")

if __name__ == "__main__":
    ref = 'references/IMG_0392_inputs_vis.jpg'
    out = 'TestSamples/examples/results/IMG_0392_inputs_vis.jpg'
    compare_images(ref, out)
