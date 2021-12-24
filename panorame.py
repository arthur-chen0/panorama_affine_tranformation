import cv2
import numpy as np

p1 = cv2.imread('/d/computer_vision/1.bmp')
p2 = cv2.imread('/d/computer_vision/2.bmp')
print('p1:', p1.shape)
print('p2:', p2.shape)

offset = 50

target = np.array(
    [[117, 163 + offset, 1, 0, 0, 0], 
     [0, 0, 0, 117, 163 + offset, 1], 
     [22, 256 + offset, 1, 0, 0, 0], 
     [0, 0, 0, 22, 256 + offset, 1],
     [155, 203 + offset, 1, 0, 0, 0], 
     [0, 0, 0, 155, 203 + offset, 1]])
src = np.array([326,167,233,261,367,207])

param = np.linalg.solve(target,src)  # 解聯立方程 - Affine Transformation
print(param)

panorama = np.zeros((p1.shape[0] + 100, p1.shape[1] + p2.shape[1], 3))
panorama[offset:(p1.shape[0]+offset), :p1.shape[1], :] = p1
cv2.imwrite("/d/computer_vision/merged_preprocess.jpg", panorama)


