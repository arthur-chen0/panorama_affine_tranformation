import cv2
import numpy as np
import math
import pathlib

def stitch(p1, p2, param):

    panorama = np.zeros((p1.shape[0], p1.shape[1] + p2.shape[1], 3))
    panorama[:p1.shape[0], :p1.shape[1], :] = p1

    intersection = np.array([p1.shape[1],0,1,0,0,0])
    ipoint = np.matmul(intersection, param)
    ipoint = math.floor(ipoint)
    print(ipoint)
    
    for y in range(p1.shape[0]):
        for x in range(p1.shape[1] + p2.shape[1]):

            transform = np.array([[x,y,1,0,0,0], [0,0,0,x,y,1]])
            target = np.matmul(transform, param)
            

            if target[0] > 0 and target[0] < (p2.shape[1] -1) and target[1] > 0 and target[1] < (p2.shape[0] - 1):

                x1 = math.floor(target[0])
                y1 = math.floor(target[1])
                x2 = math.ceil(target[0])
                y2 = math.ceil(target[1])                                                                                   
                
                topLeft = p2[y1, x1]
                topRight = p2[y1, x2]
                belowLeft = p2[y2, x1]
                belowRight = p2[y2, x2]

                a = target[0] - x1
                b = target[1] - y1
                c = (1 - a)
                d = (1 - b)
                
                point_r = (c * d * topLeft[0]) + (a * d * topRight[0]) + (c * b * belowLeft[0]) + (a * b * belowRight[0])
                point_g = (c * d * topLeft[1]) + (a * d * topRight[1]) + (c * b * belowLeft[1]) + (a * b * belowRight[1])
                point_b = (c * d * topLeft[2]) + (a * d * topRight[2]) + (c * b * belowLeft[2]) + (a * b * belowRight[2])
                    

                if panorama[y, x, 0] > 0 or panorama[y, x, 1] > 0 or panorama[y, x, 2] > 0 :  

                    weight =  (x - ipoint) / (p1.shape[1] - ipoint)  
                    
                    panorama[y, x, 0] = int(panorama[y, x, 0] * (1-weight) + int(point_r) * weight) 
                    panorama[y, x, 1] = int(panorama[y, x, 1] * (1-weight) + int(point_g) * weight) 
                    panorama[y, x, 2] = int(panorama[y, x, 2] * (1-weight) + int(point_b) * weight)  
                else:
                    panorama[y, x, 0] = int(point_r)
                    panorama[y, x, 1] = int(point_g)
                    panorama[y, x, 2] = int(point_b)
    panorama = panorama[:, 0:panorama.shape[1] - ipoint, :]
    return panorama



if __name__ == "__main__":

    path = pathlib.Path(__file__).parent.resolve()
    print(path)

    p1 = cv2.imread(str(path) + '/image/1.bmp')
    p2 = cv2.imread(str(path) + '/image/2.bmp')
    p3 = cv2.imread(str(path) + "/image/3.bmp")
    print('p1:', p1.shape)
    print('p2:', p2.shape)
    print('p3:', p3.shape )


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

    panorama = stitch(p1, p2, param)

    panorama = panorama.astype(np.uint8)
  
    cv2.imwrite(str(path) + "/out/panorama_1.jpg",panorama)

    print("===============================================================")

    panorama_1 = cv2.imread(str(path) + '/out/panorama_1.jpg')

    target = np.array(
        [[451, 162, 1, 0, 0, 0], 
        [0, 0, 0, 451, 162, 1], 
        [555, 173, 1, 0, 0, 0], 
        [0, 0, 0, 555, 173, 1],
        [462, 292, 1, 0, 0, 0], 
        [0, 0, 0, 462, 292, 1]])
    src = np.array([28,152,130,167,32,285])

    param = np.linalg.solve(target,src)
    print(param)

    panorama = stitch(panorama_1, p3, param)

    panorama = panorama.astype(np.uint8)
  
    cv2.imwrite(str(path) + "/out/panorama_2.jpg",panorama)