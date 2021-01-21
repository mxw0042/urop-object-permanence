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
lower_yellow = np.array([30, 90, 90]) 
upper_yellow = np.array([40, 255, 255])

#Make a window
window = Tk()
window.title("Drag & Drop")
main_window = Window(window)

#Create a circle object from the DragCircle class
circle = DragCircle(main_window, 100, 100, "yellow")
circle2 = DragCircle(main_window, 200, 200, "green")
circle3 = DragCircle(main_window, 300, 300, "blue")
cup= DragCup(main_window, 50, 200)

meas=[[],[],[]]
pred=[[(450, 450)], [(300, 300)], [(170, 170)]]
frame_kalman = np.zeros((400,600,3), np.uint8) # drawing canvas

cv2.namedWindow("kalman")
cv2.moveWindow("kalman", 800,300) 

error_cov = np.array([np.eye(4)*1000.0, np.eye(4)*1000.0, np.eye(4)*1000.0])

x = np.array([[np.float32(0), np.float32(0), 0.5*np.pi, 0.0],
            [np.float32(0), np.float32(0), 0.5*np.pi, 0.0],
            [np.float32(0), np.float32(0), 0.5*np.pi, 0.0]])

prev_time=time.time()

# plt.figure()
# plt.gca().invert_yaxis()


def paint(c):
    global frame_kalman,meas,pred, kalman
    
    meas_colors=[(150,0,0), (0,100,0), (0,100,100)]
    pred_colors=[(300,0,0), (0,300,0), (0,300,300)]

    for i in range(len(meas[c])-1): 
        cv2.line(frame_kalman,meas[c][i],meas[c][i+1],meas_colors[c]) #dark 
    for i in range(len(pred[c])-1): 
        cv2.line(frame_kalman,pred[c][i],pred[c][i+1],pred_colors[c]) #bright


#Start the animation loop
def task(prev_time, error_cov, count, x):
    x2=window.winfo_rootx()+main_window.canvas.winfo_x()
    y2=window.winfo_rooty()+main_window.canvas.winfo_y()
    x1=x2+main_window.canvas.winfo_width()+170
    y1=y2+main_window.canvas.winfo_height()+150
    
    N = 1000
    X = np.linspace(x1, x2, N)
    Y = np.linspace(y1, y2, N)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (4,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    img=ImageGrab.grab((x2,y2,x1,y1)).save("test_kalman.png")
    frame = cv2.imread("test_kalman.png")
    
    # It converts the BGR color space of image to HSV color space 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    # preparing the mask to overlay 
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
    mask_green = cv2.inRange(hsv, lower_green, upper_green) 
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) 
        
    # The black region in the mask has the value of 0, 
    # so when multiplied with original image removes all non-blue regions 
    result = cv2.bitwise_or(cv2.bitwise_or(mask_blue, mask_green), mask_yellow) 

    masks=[mask_blue, mask_green, mask_yellow]
    colors=["Blue", "Green", "Yellow"]
    for i in range(len(masks)):
        M=cv2.moments(masks[i])
        if M["m00"]!=0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            meas[i].append((cX,cY))
            mp = [cX,cY]
            filterstep=time.time()-prev_time
            prev_time=time.time()

            x[i], error_cov[i] = ekf(mp, x[i], filterstep, error_cov[i], count)
            pred[i].append((int(x[i][0]),int(x[i][1])))
            paint(i)

            # F = multivariate_normal(np.array(x[i]), error_cov[i])
            # Z = F.pdf(pos)
            # plt.contourf(X, Y, Z, cmap=cm.viridis)
    plt.show(block=False)

    cv2.imshow("kalman",frame_kalman)
    count+=1
    window.after(10, task, prev_time, error_cov, count, x)  # reschedule event 
    
window.after(500, task, prev_time, error_cov, 0, x)
window.mainloop()

# Closes all the frames 
cv2.destroyAllWindows() 