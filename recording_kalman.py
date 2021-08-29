from tkinter import *
import time
import cv2  
import numpy as np 
import matplotlib.pyplot as plt
import math
from matplotlib.colors import hsv_to_rgb
from PIL import ImageGrab
from scipy.stats import multivariate_normal
from matplotlib import cm

from cup_game import Window, DragCircle, DragCup
from ekf import ekf
from statistics import mean

cap=cv2.VideoCapture('sample_game.mp4')

# Threshold of green in HSV space 
lower_green = np.array([50, 100, 100]) 
upper_green = np.array([70, 255, 255]) 

# Threshold of blue in HSV space 
lower_blue = np.array([0, 150, 210]) #np.array([110,150,100])
upper_blue = np.array([10, 255, 255]) #np.array([130,255,255])

# Threshold of yellow in HSV space 
lower_yellow = np.array([80, 100, 100]) 
upper_yellow = np.array([100, 255, 255])

# Threshold of red in HSV space 
lower_red = np.array([110,150,150]) #np.array([155,25,0])
upper_red = np.array([130,255,255]) #np.array([179,255,255])


meas=[[],[],[],[]]
pred=[[(301, 330)], [(200, 230)], [(101, 131)], [(51, 228)]]
error=[[], [], [], []]
error_kl=[[], [], [], []]

frame_kalman = np.zeros((400,600,3), np.uint8) # drawing canvas
covered=[4, 4, 4, 4] #4 is uncovered
cX=[0, 0, 0, 0]
cY=[0, 0, 0, 0]
x = np.array([[np.float32(0), np.float32(0), 0.5*np.pi, 0.0],
            [np.float32(0), np.float32(0), 0.5*np.pi, 0.0],
            [np.float32(0), np.float32(0), 0.5*np.pi, 0.0], 
            [np.float32(0), np.float32(0), 0.5*np.pi, 0.0]])

cv2.namedWindow("kalman")
cv2.moveWindow("kalman", 800,300) 

kalman1 = cv2.KalmanFilter(4,2)
kalman1.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman1.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman1.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.001
kalman1.errorCovPost =np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)

kalman2 = cv2.KalmanFilter(4,2)
kalman2.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman2.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman2.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.001
kalman2.errorCovPost =np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)

kalman3 = cv2.KalmanFilter(4,2)
kalman3.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman3.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman3.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.001
kalman3.errorCovPost =np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)

kalman4 = cv2.KalmanFilter(4,2)
kalman4.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
kalman4.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
kalman4.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.001
kalman4.errorCovPost =np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32)

k=[kalman1, kalman2, kalman3, kalman4]
# kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * 0.00003
outF = open("predictions.txt", "w")
count=0

def bhattacharyya(mean1, cov1, mean2, cov2):
    cov=(1/2)*(cov1+cov2)
    t1=(1/8)*np.sqrt((mean1-mean2)@np.linalg.inv(cov)@(mean1-mean2).T)
    t2=(1/2)*np.log(np.linalg.det(cov)/np.sqrt(np.linalg.det(cov1)*np.linalg.det(cov2)))
    return t1+t2

def distance_kullback(A, B):
    """Kullback leibler divergence between two covariance matrices A and B.
    :param A: First covariance matrix
    :param B: Second covariance matrix
    :returns: Kullback leibler divergence between A and B
    """
    dim = A.shape[0]
    logdet = np.log(np.linalg.det(B) / np.linalg.det(A))
    kl = np.trace(np.dot(np.linalg.inv(B), A)) - dim + logdet
    return 0.5 * kl

def paint(c):
    global frame_kalman,meas,pred, kalman
    meas_colors=[(150,0,0), (0,100,0), (0,100,100), (0, 0, 100)]
    pred_colors=[(300,0,0), (0,300,0), (0,300,300), (0, 0, 300)]

    for i in range(len(meas[c])-1): 
        cv2.line(frame_kalman,meas[c][i],meas[c][i+1],meas_colors[c]) #dark 
    for i in range(len(pred[c])-1): 
        cv2.line(frame_kalman,pred[c][i],pred[c][i+1],pred_colors[c]) #bright

    # chi2inv(0.95, 2) = 5.9915
    vals, vecs = np.linalg.eigh(5.9915 * k[c].errorCovPost)
    indices = vals.argsort()[::-1]
    vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

    axes = int(vals[0] + .5), int(vals[1] + .5)
    angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
    cv2.ellipse(frame_kalman, pred[c][i+1], axes, angle, 0, 360, meas_colors[c], 2)


distance_from_groundtruth=[]
trace_error=[]
kl_error=[]
#Start the animation loop
while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('frame1',frame)
    # It converts the RGB color space of image to HSV color space 
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)


    # preparing the mask to overlay 
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
    mask_green = cv2.inRange(hsv, lower_green, upper_green) 
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) 
    mask_red = cv2.inRange(hsv, lower_red, upper_red) 
    masks=[mask_blue, mask_green, mask_yellow, mask_red]

    if (ret):
        for c in range(len(masks)):
            M=cv2.moments(masks[c])
            contours,hierarchy = cv2.findContours(masks[c].copy(), 1, 2)
            if len(contours)!=0:
                covered[c]=4
                area=cv2.contourArea(contours[0])
                if M["m00"]!=0 and (area>1500 or area==0 or (area>100 and c==3)):
                    cX[c] = int(M["m10"] / M["m00"])
                    cY[c] = int(M["m01"] / M["m00"])
                    mp = np.array([[np.float32(cX[c])],[np.float32(cY[c])]])
                    meas[c].append((cX[c],cY[c]))
                    k[c].correct(mp)
                prev_cov=k[c].errorCovPost.copy()
                tp = k[c].predict()
                if count>16:
                    pred[c].append((int(tp[0]),int(tp[1])))
                
                    error[c].append(np.trace(k[c].errorCovPost))
                    error_kl[c].append(distance_kullback(prev_cov, k[c].errorCovPost))
            else: 
                if covered[c]==4:
                    distances=dict()
                    for i in range(len(masks)):
                        if i!=c and np.any(k[i].errorCovPost): 
                            distances[i]=bhattacharyya(x[c], k[c].errorCovPost, x[i], k[i].errorCovPost)
                    covered[c]=min(distances, key=distances.get)
                    meas[c].append(meas[covered[c]][-1])
                    cX[c] = meas[covered[c]][-1][0]
                    cY[c] = meas[covered[c]][-1][1]
            
            
            if count>16:
                data="{}: actual- {} \t predicted- {} \t error (trace)- {} \t error (kl)- {}".format(c, (cX,cY), (int(tp[0]),int(tp[1])), error[c][-1], error_kl[c][-1])
                outF.write(data)
                outF.write("\n")
                distance_from_groundtruth+=[math.hypot(float(tp[0][0])-cX[c], float(tp[1][0])-cY[c])]
                trace_error+=[error[c][-1]]
                kl_error+=[error_kl[c][-1]]
                paint(c)
    if count>16:
        print("eucildean: ", mean(distance_from_groundtruth))      
        print("trace error: ", mean(trace_error))   
        print("kl error: ", mean(kl_error))
    count+=1
    cv2.imshow("kalman",frame_kalman)
    if cv2.waitKey(35) & 0xFF == ord('q'):
        break
    

cap.release()
# Closes all the frames 
cv2.destroyAllWindows() 