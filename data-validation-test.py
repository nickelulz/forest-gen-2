import os, sys, cv2
from matplotlib import pyplot as plt
import numpy as np

STAND_IMAGE_PATH = "./data/forest-original/test/images/stand_1809.png"
STAND_LABEL_PATH = "./data/forest-original/test/labels/stand_1809.txt"
IMAGE_SIZE = 640

def main():
    image = cv2.imread(STAND_IMAGE_PATH)
    image_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with open(STAND_LABEL_PATH, 'r') as labels_file:
        for line in labels_file.readlines():
            indices = np.array(list(map(float, line.strip().split()))[1:], dtype=np.float64) * IMAGE_SIZE
            polygon = np.array(indices, dtype=np.int32).reshape(-1, 2) 
            print(len(polygon), len(indices))
            cv2.polylines(image_cv, [polygon], isClosed=True, color=(0, 255, 0), thickness=1)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_cv)
    plt.axis("off")
    plt.show()

if __name__ == '__main__':
    main()
