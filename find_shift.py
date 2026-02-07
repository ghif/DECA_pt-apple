
import cv2
import numpy as np

def find_best_shift():
    img1 = cv2.imread('debug_output/4_shape.jpg', 0)
    img2 = cv2.imread('debug_output/5_detail.jpg', 0)
    
    if img1 is None or img2 is None:
        return

    best_mse = float('inf')
    best_shift = (0, 0)
    
    # Check +/- 5 pixels
    h, w = img1.shape
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            y1 = max(0, dy); y2 = min(h, h + dy)
            x1 = max(0, dx); x2 = min(w, w + dx)
            
            y1_r = max(0, -dy); y2_r = min(h, h - dy)
            x1_r = max(0, -dx); x2_r = min(w, w - dx)
            
            crop1 = img1[y1:y2, x1:x2]
            crop2 = img2[y1_r:y2_r, x1_r:x2_r]
            
            mse = np.mean((crop1 - crop2)**2)
            if mse < best_mse:
                best_mse = mse
                best_shift = (dx, dy)
                
    print(f"Best MSE: {best_mse:.4f} at Shift (dx={best_shift[0]}, dy={best_shift[1]})")

if __name__ == '__main__':
    find_best_shift()
