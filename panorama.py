import cv2
import numpy as np
import math

p1 = cv2.imread('/d/computer_vision/hw_panorama/1.bmp')
p2 = cv2.imread('/d/computer_vision/hw_panorama/2.bmp')
print('p1:', p1.shape)
print('p2:', p2.shape)


target = np.array(
    [[326, 167, 1, 0, 0, 0], 
     [0, 0, 0, 326, 167, 1], 
     [233, 261, 1, 0, 0, 0], 
     [0, 0, 0, 233, 261, 1],
     [367, 207, 1, 0, 0, 0], 
     [0, 0, 0, 367, 207, 1]])
src = np.array([117,163,22,256,155,203])

param = np.linalg.solve(target,src)
print(param)

panorama = np.zeros((p1.shape[0], p1.shape[1] + p2.shape[1], 3))
panorama[:p1.shape[0], :p1.shape[1], :] = p1
cv2.imwrite("/d/computer_vision/panorama_preprocess.jpg", panorama)


for y in range(p1.shape[0]):
    for x in range(p1.shape[1] + p2.shape[1]):

        t = np.array([[x,y,1,0,0,0], [0,0,0,x,y,1]])
        s = np.matmul(t, param)
        

        if s[0] > 0 and s[0] < (p2.shape[1] -1) and s[1] > 0 and s[1] < (p2.shape[0] - 1):

            x1 = math.floor(s[0])
            y1 = math.floor(s[1])
            x2 = math.ceil(s[0])
            y2 = math.ceil(s[1])                                                                                   
            
            # 四點顏色，top-left, top-right, bottom-left, bottem-right
            topLeft = p2[y1, x1]
            topRight = p2[y1, x2]
            belowLeft = p2[y2, x1]
            belowRight = p2[y2, x2]

            a = s[0] - x1
            b = s[1] - y1
            c = (1 - a)
            d = (1 - b)
            
            # Binear Interpolation 顏色計算 - 3 channel
            point_r = (c * d * topLeft[0]) + (a * d * topRight[0]) + (c * b * belowLeft[0]) + (a * b * belowRight[0])
            point_g = (c * d * topLeft[1]) + (a * d * topRight[1]) + (c * b * belowLeft[1]) + (a * b * belowRight[1])
            point_b = (c * d * topLeft[2]) + (a * d * topRight[2]) + (c * b * belowLeft[2]) + (a * b * belowRight[2])
                

            if panorama[y, x, 0] > 0 or panorama[y, x, 1] > 0 or panorama[y, x, 2] > 0 :  

                weight =  (x-156) / (384-156)  
                
                panorama[y, x, 0] = int(panorama[y, x, 0] * (1-weight) + int(point_r) * weight) 
                panorama[y, x, 1] = int(panorama[y, x, 1] * (1-weight) + int(point_g) * weight) 
                panorama[y, x, 2] = int(panorama[y, x, 2] * (1-weight) + int(point_b) * weight)  
            else:
                panorama[y, x, 0] = int(point_r)
                panorama[y, x, 1] = int(point_g)
                panorama[y, x, 2] = int(point_b)

panorama = panorama.astype(np.uint8)
# merged_corp = merged[14:864, :, :]

# minInColumns = np.amin(merged[:,:,0], axis=0)

# cv2.imshow("res", panorama)            
cv2.imwrite("panorama_A.jpg",panorama)