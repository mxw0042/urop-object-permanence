from tkinter import *
import time
import cv2  
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import ImageGrab
from scipy.stats import multivariate_normal
from matplotlib import cm

from cup_game import Window, DragCircle, DragCup
from ekf import ekf

# Threshold of green in HSV space 
lower_green = np.array([50, 100, 100]) 
upper_green = np.array([70, 255, 255]) 

# Threshold of blue in HSV space 
lower_blue = np.array([110,200,200])
upper_blue = np.array([130,255,255])

# Threshold of yellow in HSV space 
lower_yellow = np.array([30, 100, 100]) 
upper_yellow = np.array([40, 255, 255])

# Threshold of red in HSV space 
lower_red = np.array([155,25,0])
upper_red = np.array([179,255,255])

#Make a window
window = Tk()
window.title("Drag & Drop")
main_window = Window(window)

#Create a circle object from the DragCircle class
circle = DragCircle(main_window, 100, 100, "yellow")
circle2 = DragCircle(main_window, 200, 200, "green")
circle3 = DragCircle(main_window, 300, 300, "blue")
cup= DragCup(main_window, 50, 200)

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
# plt.figure()
# plt.gca().invert_yaxis()

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



#Start the animation loop
def task(prev_time, error_cov, count, x):
    x2=window.winfo_rootx()+main_window.canvas.winfo_x()
    y2=window.winfo_rooty()+main_window.canvas.winfo_y()
    x1=x2+main_window.canvas.winfo_width()
    y1=y2+main_window.canvas.winfo_height()
    
    N = 1000
    X = np.linspace(x1, x2, N)
    Y = np.linspace(y1, y2, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (4,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    # It converts the RGB color space of image to HSV color space 
    img=ImageGrab.grab((x2,y2,x1,y1))
    frame = np.array(img)
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV) 

    # preparing the mask to overlay 
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
    mask_green = cv2.inRange(hsv, lower_green, upper_green) 
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) 
    mask_red = cv2.inRange(hsv, lower_red, upper_red) 

    masks=[mask_blue, mask_green, mask_yellow, mask_red]

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
            data="{}: actual- {} \t predicted- {} \t error (trace)- {} \t error (kl)- {}".format(c, (cX[c],cY[c]), (int(x[c][0]),int(x[c][1])), error[c][-1], error_kl[c][-1])
            outF.write(data)
            outF.write("\n")
            
            paint(c)
            # F = multivariate_normal(np.array(x[c]), error_cov[c])
            # Z = F.pdf(pos)
            # plt.contourf(X, Y, Z, cmap=cm.viridis)
            # plt.show(block=False)
        else: 
            if covered[c]==4:
                distances=dict()
                for i in range(len(masks)):
                    if i!=c: 
                        distances[i]=bhattacharyya(x[c], error_cov[c], x[i], error_cov[i])
                covered[c]=min(distances, key=distances.get)
            prev_cov=error_cov[c].copy()
            error_cov[c]*=1.05
            pred[c].append((int(x[covered[c]][0]),int(x[covered[c]][1])))
            error[c].append(np.trace(error_cov[c]))
            error_kl[c].append(distance_kullback(prev_cov, error_cov[c]))
            data="{}: actual- {} \t predicted- {} \t error (trace)- {} \t error (kl)- {}".format(c, (cX[c],cY[c]), (int(x[c][0]),int(x[c][1])), error[c][-1], error_kl[c][-1])
            outF.write(data)
            outF.write("\n")
            
            paint(c)

    cv2.imshow("kalman",frame_kalman)
    count+=1
    window.after(10, task, prev_time, error_cov, count, x)  # reschedule event 
    
window.after(500, task, prev_time, error_cov, 0, x)
window.mainloop()

# Closes all the frames 
cv2.destroyAllWindows() 