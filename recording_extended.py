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
from scipy.spatial import distance

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
pred=[[], [], [], []]
error=[[], [], [], []]
error_kl=[[], [], [], []]

frame_kalman = np.zeros((400,600,3), np.uint8) # drawing canvas

cv2.namedWindow("kalman")
cv2.moveWindow("kalman", 800,300) 

error_cov = np.array([np.eye(4)*0, np.eye(4)*0, np.eye(4)*0, np.eye(4)*0])

x = np.array([[np.float32(0), np.float32(0), 0.5*np.pi, 0.0],
            [np.float32(0), np.float32(0), 0.5*np.pi, 0.0],
            [np.float32(0), np.float32(0), 0.5*np.pi, 0.0], 
            [np.float32(0), np.float32(0), 0.5*np.pi, 0.0]])

cX=[0, 0, 0, 0]
cY=[0, 0, 0, 0]

prev_time=[time.time(), time.time(), time.time(), time.time()]
filterstep=[0, 0, 0, 0]
covered=[4, 4, 4, 4] #4 is uncovered

count=0

outF = open("predictions.txt", "w")

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
        if c==1:
            cv2.line(frame_kalman,pred[c][i],pred[c][i+1],pred_colors[c]) #bright
        else:
            cv2.line(frame_kalman,pred[c][i],pred[c][i+1],pred_colors[c]) #bright

    vals, vecs = np.linalg.eigh(5.9915 * error_cov[c])  # chi2inv(0.95, 2) = 5.9915
    indices = vals.argsort()[::-1]
    vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

    axes = int(vals[0]), int(vals[1])
    angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
    cv2.ellipse(frame_kalman, pred[c][len(pred[c])-1], axes, angle, 0, 360, meas_colors[c], 2)

mahalanobis=[]
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
                    meas[c].append((cX[c],cY[c]))
                filterstep[c]=time.time()-prev_time[c]
                prev_time[c]=time.time()
                prev_cov=error_cov[c].copy()
                
                x[c], error_cov[c] = ekf([cX[c],cY[c]], x[c], filterstep[c], error_cov[c], count)
                pred[c].append((int(x[c][0]),int(x[c][1])))
                error[c].append(np.trace(error_cov[c]))
                error_kl[c].append(distance_kullback(prev_cov, error_cov[c]))
            else: 
                if covered[c]==4:
                    distances=dict()
                    for i in range(len(masks)):
                        if i!=c and np.any(error_cov[i]): 
                            distances[i]=bhattacharyya(x[c], error_cov[c], x[i], error_cov[i])
                    covered[c]=min(distances, key=distances.get)
                    meas[c].append(meas[covered[c]][-1])
                    cX[c] = meas[covered[c]][-1][0]
                    cY[c] = meas[covered[c]][-1][1]

            data="{}: actual- {} \t predicted- {} \t error (trace)- {} \t error (kl)- {}".format(c, (cX[c],cY[c]), (int(x[c][0]),int(x[c][1])), error[c][-1], error_kl[c][-1])
            outF.write(data)
            outF.write("\n")
            if count>5:
                inv=np.linalg.inv(np.array(error_cov[c])[:2,:2])
                mahalanobis+=[distance.mahalanobis([cX[c],cY[c]], [float(x[c][0]), float(x[c][1])], inv)]
                distance_from_groundtruth+=[math.hypot(float(x[c][0]-cX[c]), float(x[c][1]-cY[c]))]
                trace_error+=[error[c][-1]]
                kl_error+=[error_kl[c][-1]]
            paint(c)

    if count>5:
        print("mahalanobis: ", mean(mahalanobis))
        print("eucildean: ", mean(distance_from_groundtruth))
        print("trace error: ", mean(trace_error))   
        print("kl error: ", mean(kl_error))
    cv2.imshow("kalman",frame_kalman)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    count+=1
    

cap.release()
# Closes all the frames 
cv2.destroyAllWindows() 