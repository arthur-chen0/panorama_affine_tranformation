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

# t = np.array([[383,512,1,0,0,0], [0,0,0,383,512,1]])
# s = np.matmul(t, param)
# print(s)

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
            
            # print(x, " ", y, " ", x1, " ", x2, " ", y1, " ", y2)
                                                                                                                                
            # topLeft = p2[y1, x1]
            # topRight = p2[y1, x2]
            # belowLeft = p2[y2, x1]
            # belowRight = p2[y2, x2]

            # a = s[0] - x1
            # b = s[1] - y1
            # c = (1 - a) # x2 - s[0]
            # d = (1 - b) # y2 - s[1]

            # point_r = (topLeft[0] * c * d) + (topRight[0] * a * d) + (belowLeft[0] * b * c) + (belowRight[0] * a * b)
            # point_g = (topLeft[1] * c * d) + (topRight[1] * a * d) + (belowLeft[1] * b * c) + (belowRight[1] * a * b)
            # point_b = (topLeft[2] * c * d) + (topRight[2] * a * d) + (belowLeft[2] * b * c) + (belowRight[2] * a * b)


            
            # 四點顏色，top-left, top-right, bottom-left, bottem-right
            tlp = p2[y1, x1]
            trp = p2[y1, x2]
            blp = p2[y2, x1]
            brp = p2[y2, x2]
            
            # Binear Interpolation 顏色計算 - 3 channel
            point_r = (x2-s[0])*(y2-s[1])*tlp[0] + \
                (s[0]-x1)*(y2-s[1])*trp[0] +  \
                (x2-s[0])*(s[1]-y1)*blp[0] + \
                (s[0]-x1)*(s[1]-y1)*brp[0] 
                
            point_g = (x2-s[0])*(y2-s[1])*tlp[1] + \
                (s[0]-x1)*(y2-s[1])*trp[1] +  \
                (x2-s[0])*(s[1]-y1)*blp[1] + \
                (s[0]-x1)*(s[1]-y1)*brp[1] 
                
            point_b = (x2-s[0])*(y2-s[1])*tlp[2] + \
                (s[0]-x1)*(y2-s[1])*trp[2] +  \
                (x2-s[0])*(s[1]-y1)*blp[2] + \
                (s[0]-x1)*(s[1]-y1)*brp[2] 
            

            # if panorama[y, x, 0] > 0 or panorama[y, x, 1] > 0 or panorama[y, x, 2] > 0:
            #     weight =  (x-156) / (384-156)  
                
                
            #     panorama[y, x, 0] = int(panorama[y, x, 0] * (1-weight) + int(point_r) * weight)  
            #     panorama[y, x, 1] = int(panorama[y, x, 1] * (1-weight) + int(point_g) * weight) 
            #     panorama[y, x, 2] = int(panorama[y, x, 2] * (1-weight) + int(point_b) * weight) 
            # else:
            #     panorama[y, x, 0] = int(point_r)  # 未在交集內，直接用 binear interpolation 的顏色
            #     panorama[y, x, 1] = int(point_g)
            #     panorama[y, x, 2] = int(point_b) 

            if panorama[y, x, 0] > 0 or panorama[y, x, 1] > 0 or panorama[y, x, 2] > 0 :   # 3 channel 有一維不是0，表是是交集處。此處有點鳥，應該有更好的判斷方法。
            
                # print(x)
                # break

                # weight = 0.5      # 交集處各取一半
                weight =  (x-156) / (384-156)  # blending 計算，這裡也有點鳥，寫死了，手工找了交集範圍
                # weight =  (x-812) / (1280-812) * 0.8    # for testing
                # weight =  (x-812) / (1280-812) * 0.5    # for testing
                
                panorama[y, x, 0] = int(panorama[y, x, 0] * (1-weight) + int(point_r) * weight)  # blending 計算
                panorama[y, x, 1] = int(panorama[y, x, 1] * (1-weight) + int(point_g) * weight)  # blending 計算
                panorama[y, x, 2] = int(panorama[y, x, 2] * (1-weight) + int(point_b) * weight)  # blending 計算
            else:
                panorama[y, x, 0] = int(point_r)  # 未在交集內，直接用 binear interpolation 的顏色
                panorama[y, x, 1] = int(point_g)
                panorama[y, x, 2] = int(point_b) 

panorama = panorama.astype(np.uint8)
# merged_corp = merged[14:864, :, :]

# minInColumns = np.amin(merged[:,:,0], axis=0)

# cv2.imshow("res", panorama)            
cv2.imwrite("panorama_A.jpg",panorama)