import cv2
import numpy as np

def main():
    img = cv2.imread('references/IMG_0392_inputs_vis.jpg')
    h, w, c = img.shape
    panel_w = w // 6
    for i in range(6):
        panel = img[:, i*panel_w:(i+1)*panel_w]
        mean = np.mean(panel, axis=(0, 1))
        std = np.std(panel, axis=(0, 1))
        print(f"Panel {i+1} | Mean BGR: {mean} | Std: {std}")

if __name__ == '__main__':
    main()
