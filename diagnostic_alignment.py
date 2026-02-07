
import cv2
import numpy as np
import os

def check_alignment():
    shape_path = 'debug_output/4_shape.jpg'
    detail_path = 'debug_output/5_detail.jpg'
    
    if not os.path.exists(shape_path) or not os.path.exists(detail_path):
        print("Missing debug files.")
        return

    img_shape = cv2.imread(shape_path).astype(np.float32)
    img_detail = cv2.imread(detail_path).astype(np.float32)
    
    if img_shape.shape != img_detail.shape:
        print(f"Shape mismatch: {img_shape.shape} vs {img_detail.shape}")
        return

    mse = np.mean((img_shape - img_detail)**2)
    mean_intensity_shape = np.mean(img_shape)
    mean_intensity_detail = np.mean(img_detail)
    
    print(f"MSE: {mse:.4f}")
    print(f"Mean Intensity Shape: {mean_intensity_shape:.2f}")
    print(f"Mean Intensity Detail: {mean_intensity_detail:.2f}")
    
    if mse < 10: # Reasonable threshold for alignment with some texture
        print("Alignment looks GOOD.")
    else:
        print("Alignment or shading difference is SIGNIFICANT.")

if __name__ == '__main__':
    check_alignment()
