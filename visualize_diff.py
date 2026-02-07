
import cv2
import numpy as np

def visualize_diff():
    img1 = cv2.imread('debug_output/4_shape.jpg')
    img2 = cv2.imread('debug_output/5_detail.jpg')
    
    if img1 is None or img2 is None:
        return

    diff = cv2.absdiff(img1, img2)
    # Enhance diff for visibility
    diff_vis = np.clip(diff.astype(np.float32) * 10, 0, 255).astype(np.uint8)
    
    # Create side-by-side
    combined = np.hstack([img1, img2, diff_vis])
    cv2.imwrite('debug_output/diff_analysis.jpg', combined)
    print("Saved debug_output/diff_analysis.jpg")

if __name__ == '__main__':
    visualize_diff()
