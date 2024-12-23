import cv2
import numpy as np
image1 = cv2.imread('863.jpg')[:,0:100]
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2 = cv2.imread('864.jpg')[:,0:100]
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
print(np.max(image1), np.min(image1))
print(np.max(image2), np.min(image2))