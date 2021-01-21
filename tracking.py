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

meas=[]
pred=[(0, 0)]
frame_kalman = np.zeros((400,600,3), np.uint8) # drawing canvas

cv2.namedWindow("kalman")
cv2.moveWindow("kalman", 800,300) 

error_cov = np.eye(4)*1000.0
x = np.array([np.float32(0), np.float32(0), 0.5*np.pi, 0.0])

prev_time=time.time()

# plt.figure()
# plt.gca().invert_yaxis()


def paint():
    global frame_kalman,meas,pred, kalman

    for i in range(len(meas)-1): 
        cv2.line(frame_kalman,meas[i],meas[i+1],(0,100,0)) #green
    for i in range(len(pred)-1): 
        cv2.line(frame_kalman,pred[i],pred[i+1],(0,0,200)) #red


#Start the animation loop
def task(prev_time, error_cov, count, x):
    x2=window.winfo_rootx()+main_window.canvas.winfo_x()
    y2=window.winfo_rooty()+main_window.canvas.winfo_y()
    x1=x2+main_window.canvas.winfo_width()+170
    y1=y2+main_window.canvas.winfo_height()+150
    
    N = 60
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
          if i==0:
            meas.append((cX,cY))
            mp = [cX,cY]
            filterstep=time.time()-prev_time
            prev_time=time.time()

            x, error_cov = ekf(mp, x, filterstep, error_cov, count)
            pred.append((int(x[0]),int(x[1])))
            paint()

            # F = multivariate_normal(np.array([int(x[0]),int(x[1]), 0, 0]), error_cov)
            # Z = F.pdf(pos)
            # plt.contourf(X, Y, Z, cmap=cm.viridis)
            # plt.show(block=False)

    cv2.imshow("kalman",frame_kalman)
    count+=1
    window.after(10, task, prev_time, error_cov, count, x)  # reschedule event 
    
window.after(500, task, prev_time, error_cov, 0, x)
window.mainloop()

# Closes all the frames 
cv2.destroyAllWindows() 