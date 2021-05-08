import csv
import numpy as np
from spatialmath import *
from spatialmath.pose3d import * 
from math import pi

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import statistics

filename = "tf (2).csv"

fields = []
rows = []
pose = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)

    for row in csvreader: 
        rows.append(row)

    print("Total no. of rows: %d"%(csvreader.line_num))


Xr=[]
Yr=[]
Zr=[]
dt_array=[]

print("starting to read measurements...")

# objs = {
#   '"obj-10"': axes_10,
#   '"obj-7"': axes_7,
#   '"obj-3"': axes_3,
#   '"obj-4"': axes_4,
#   '"obj-12"': axes_12
# }
prev_row=[]

for row in rows:
    if '"obj-10"' in row:
        Xr.append(float(row[11]))
        Yr.append(float(row[12]))
        Zr.append(float(row[13]))
        if len(dt_array)==0:
            time_elapsed=0.03334887
        else: 
            time_elapsed=(float(row[5])+float(row[6])*(10**(-9)))-(float(prev_row[5])+float(prev_row[6])*(10**-9))
        dt_array.append(time_elapsed)
        prev_row=row.copy()


measurements = np.vstack((Xr,Yr,Zr))
m = measurements.shape[1]
print(measurements.shape)
x = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]).T

xt = []
yt = []
zt = []
  
P = 100.0*np.eye(9)
# dt = 1.62*10**(-9) # Time Step between Filter Steps



H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],           #measurement matrix
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

rp = 1.0**2  # Noise of Position Measurement
R = np.matrix([[rp, 0.0, 0.0],          #measurement noise covariance matrix
               [0.0, rp, 0.0],
               [0.0, 0.0, rp]])

sj = 0.1


B = np.matrix([[0.0],           #disturbance control matrix
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0],
               [0.0]])

u = 0.0             #control input
I = np.eye(9)       #identity matrix

for filterstep in range(m):  

    dt=dt_array[filterstep]  
    
    if dt>0.034:
        print("AHHH")

    else:
        A = np.matrix([[1.0, 0.0, 0.0, dt, 0.0, 0.0, 1/2.0*dt**2, 0.0, 0.0],    #dynamic matrix
                [0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0, 1/2.0*dt**2],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,  dt],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])


        Q = np.matrix([[(dt**6)/36, 0, 0, (dt**5)/12, 0, 0, (dt**4)/6, 0, 0],           #process noise covariance matrix
                [0, (dt**6)/36, 0, 0, (dt**5)/12, 0, 0, (dt**4)/6, 0],
                [0, 0, (dt**6)/36, 0, 0, (dt**5)/12, 0, 0, (dt**4)/6],
                [(dt**5)/12, 0, 0, (dt**4)/4, 0, 0, (dt**3)/2, 0, 0],
                [0, (dt**5)/12, 0, 0, (dt**4)/4, 0, 0, (dt**3)/2, 0],
                [0, 0, (dt**5)/12, 0, 0, (dt**4)/4, 0, 0, (dt**3)/2],
                [(dt**4)/6, 0, 0, (dt**3)/2, 0, 0, (dt**2), 0, 0],
                [0, (dt**4)/6, 0, 0, (dt**3)/2, 0, 0, (dt**2), 0],
                [0, 0, (dt**4)/6, 0, 0, (dt**3)/2, 0, 0, (dt**2)]]) *sj**2


        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        x = A*x + B*u
        
        # Project the error covariance ahead
        P = A*P*A.T + Q    
        
        
        # Measurement Update (Correction)
        # ===============================
        # Compute the Kalman Gain
        S = H*P*H.T + R
        K = (P*H.T) * np.linalg.pinv(S)

        
        # Update the estimate via z
        Z = measurements[:,filterstep].reshape(H.shape[0],1)
        y = Z - (H*x)                            # Innovation or Residual
        x = x + (K*y)
        
        # Update the error covariance
        P = (I - (K*H))*P
        
        # Save states for Plotting
        xt.append(float(x[0]))
        yt.append(float(x[1]))
        zt.append(float(x[2]))

fig = plt.figure(figsize=(16,9))
ax = fig.add_subplot(111, projection='3d')
ax.plot(xt,yt,zt, label='Kalman Filter Estimate')
ax.plot(Xr, Yr, Zr, label='Real')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Ball Trajectory estimated with Kalman Filter')

# Axis equal
max_range = np.array([max(Xr)-min(Xr), max(Yr)-min(Yr), max(Zr)-min(Zr)]).max() / 3.0
mean_x = statistics.mean(Xr)
mean_y = statistics.mean(Yr)
mean_z = statistics.mean(Zr)
ax.set_xlim(mean_x - max_range, mean_x + max_range)
ax.set_ylim(mean_y - max_range, mean_y + max_range)
ax.set_zlim(mean_z - max_range, mean_z + max_range)

plt.show()