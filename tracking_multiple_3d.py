import csv
import numpy as np
from spatialmath import *
from spatialmath.pose3d import * 
from math import pi

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pylab as plt
from scipy.spatial.transform import Rotation

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


Xr=[[],[],[],[],[]]
Yr=[[],[],[],[],[]]
Zr=[[],[],[],[],[]]
rotxr=[[],[],[],[],[]]
rotyr=[[],[],[],[],[]]
rotzr=[[],[],[],[],[]]
dt_array=[[],[],[],[],[]]

print("starting to read measurements...")

# objs = {
#   '"obj-10"': axes_10,
#   '"obj-7"': axes_7,
#   '"obj-3"': axes_3,
#   '"obj-4"': axes_4,
#   '"obj-12"': axes_12
# }
prev_row=[[],[],[],[],[]]

for row in rows:
    if '"obj-10"' in row:
        j=0
    elif '"obj-7"' in row:
        j=1
    elif '"obj-3"' in row:
        j=2
    elif '"obj-4"' in row:
        j=3
    elif '"obj-12"' in row:
        j=4

    Xr[j].append(float(row[11]))
    Yr[j].append(float(row[12]))
    Zr[j].append(float(row[13]))
    rot=Rotation.from_quat([float(row[18]), float(row[15]),float(row[16]),float(row[17])])
    rot_euler = rot.as_euler('xyz', degrees=True)
    rotxr[j].append(rot_euler[0])
    rotyr[j].append(rot_euler[1])
    rotzr[j].append(rot_euler[2])
    if len(dt_array[j])==0:
        time_elapsed=0.03334887
    else: 
        time_elapsed=(float(row[5])+float(row[6])*(10**(-9)))-(float(prev_row[j][5])+float(prev_row[j][6])*(10**-9))
    dt_array[j].append(time_elapsed)
    prev_row[j]=row.copy()

measurements=[]
measurements_r=[]
m=[]
x=[]
r=[]
P=[]
P_r=[]

for i in range(5):
    measurements.append(np.vstack((Xr[i],Yr[i],Zr[i])))
    measurements_r.append(np.vstack((rotxr[i],rotyr[i],rotzr[i])))
    m.append(measurements[i].shape[1])
    x.append(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]).T)
    r.append(np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]).T)
    P.append(100.0*np.eye(9))
    P_r.append(100.0*np.eye(9))


xt = [[],[],[],[],[]]
yt = [[],[],[],[],[]]
zt = [[],[],[],[],[]]
rxt = [[],[],[],[],[]]
ryt = [[],[],[],[],[]]
rzt = [[],[],[],[],[]]
  
# P = 100.0*np.eye(9)
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
covered=[5, 5, 5, 5, 5] #4 is uncovered

def bhattacharyya(mean1, cov1, mean2, cov2):
    cov=(1/2)*(cov1+cov2)
    t1=(1/8)*np.sqrt((mean1-mean2)@np.linalg.inv(cov)@(mean1-mean2).T)
    t2=(1/2)*np.log(np.linalg.det(cov)/np.sqrt(np.linalg.det(cov1)*np.linalg.det(cov2)))
    return t1+t2

filterstep=[0, 0, 0, 0, 0]
for row in rows:
    if '"obj-10"' in row:
        i=0
    elif '"obj-7"' in row:
        i=1
    elif '"obj-3"' in row:
        i=2
    elif '"obj-4"' in row:
        i=3
    elif '"obj-12"' in row:
        i=4

    dt=dt_array[i][filterstep[i]]  

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
    
    if dt>1:
        if covered[i]==5:
            distances=dict()
            for j in range(5):
                if i!=j: 
                    distances[j]=bhattacharyya(np.array(x[i]).T, P[i], np.array(x[j]).T, P[j])
            covered[i]=min(distances, key=distances.get)
        P[i]*=1.05
        x[i]=x[covered[i]]

    else:
        covered[i]=5
        
        # Time Update (Prediction)
        # ========================
        # Project the state ahead
        x[i] = A*x[i] + B*u
        
        # Project the error covariance ahead
        P[i] = A*P[i]*A.T + Q    
        
        
        # Measurement Update (Correction)
        # ===============================
        # Compute the Kalman Gain
        S = H*P[i]*H.T + R
        K = (P[i]*H.T) * np.linalg.pinv(S)

        
        # Update the estimate via z
        Z = measurements[i][:,filterstep[i]].reshape(H.shape[0],1)
        y = Z - (H*x[i])                            # Innovation or Residual
        x[i] = x[i] + (K*y)
        
        # Update the error covariance
        P[i] = (I - (K*H))*P[i]

    #orientation 
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    r[i] = A*r[i] + B*u
    
    # Project the error covariance ahead
    P_r[i] = A*P_r[i]*A.T + Q    
    
    
    # Measurement Update (Correction)
    # ===============================
    # Compute the Kalman Gain
    S = H*P_r[i]*H.T + R
    K = (P_r[i]*H.T) * np.linalg.pinv(S)

    
    # Update the estimate via z
    Z = measurements_r[i][:,filterstep[i]].reshape(H.shape[0],1)
    y = Z - (H*r[i])                            # Innovation or Residual
    r[i] = r[i] + (K*y)
    
    # Update the error covariance
    P_r[i] = (I - (K*H))*P_r[i]


        
    # Save states for Plotting
    xt[i].append(float(x[i][0]))
    yt[i].append(float(x[i][1]))
    zt[i].append(float(x[i][2]))

    rxt[i].append(float(r[i][0]))
    ryt[i].append(float(r[i][1]))
    rzt[i].append(float(r[i][2]))

    filterstep[i]+=1

fig = plt.figure(figsize=(14,7))
plt.title('Trajectory estimated with EKF and Occlusion Condition')
ax = fig.add_subplot(121, projection='3d')
lines=[ax.plot([], [], [], lw = 2, color='#0061fc'), ax.plot([], [], [], lw = 2, color='#07fc00'), ax.plot([], [], [], lw = 2, color='#ff8800'), ax.plot([], [], [], lw = 2, color='#ff0000'), ax.plot([], [], [], lw = 2, color='#b400f5')]
ax2 = fig.add_subplot(122, projection='3d')
dots=[ax2.scatter([], [], [], lw = 2, color='#0061fc', s=1), ax2.scatter([], [], [], lw = 2, color='#07fc00', s=1), ax2.scatter([], [], [], lw = 2, color='#ff8800', s=1), ax2.scatter([], [], [], lw = 2, color='#ff0000', s=1), ax2.scatter([], [], [], lw = 2, color='#b400f5', s=1)]


xdata = [[],[],[],[],[]]
ydata = [[],[],[],[],[]]
zdata = [[],[],[],[],[]]

xdatadot = [[],[],[],[],[]]
ydatadot = [[],[],[],[],[]]
zdatadot = [[],[],[],[],[]]

count=0

# Axis equal
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.5, 1)
ax.invert_zaxis()

# Axis equal
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(-0.5, 0.5)
ax2.set_zlim(0.5, 1)
ax2.invert_zaxis()

colors=['#0061fc', '#07fc00', '#ff8800', '#ff0000', '#b400f5']

axes_10=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]
axes_7=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]
axes_3=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]
axes_4=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]
axes_12=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]

axes_all=[axes_10, axes_7, axes_3, axes_4, axes_12]

axes_10r=[ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0)]
axes_7r=[ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0)]
axes_3r=[ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0)]
axes_4r=[ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0)]
axes_12r=[ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0), ax2.quiver(0, 0, 0, 0, 0, 0)]

axes_allr=[axes_10r, axes_7r, axes_3r, axes_4r, axes_12r]

ax_colors=['#E16F6D', '#4eb604', '#0d87c3']

def update(count):
    global axes_10, axes_7, axes_3, axes_4, axes_12, axes_10r, axes_7r, axes_3r, axes_4r, axes_12r
    for i in range(len(lines)):
        #currently plotting the same quiver 3 times, should change to one for each axis or just one overall 
        for j in range(3):
            axes_all[i][j].remove()   
            axes_all[i][j] = ax.quiver(xt[i][count], yt[i][count], zt[i][count], rxt[i][count], ryt[i][count], rzt[i][count], length=0.05, normalize=True, color=ax_colors[j])
       
            axes_allr[i][j].remove()   
            axes_allr[i][j] = ax2.quiver(Xr[i][count], Yr[i][count], Zr[i][count], rotxr[i][count], rotyr[i][count], rotzr[i][count], length=0.05, normalize=True, color=ax_colors[j])
       
        xdata[i].append(xt[i][count])
        ydata[i].append(yt[i][count])
        zdata[i].append(zt[i][count])

        xdatadot[i].append(Xr[i][count])
        ydatadot[i].append(Yr[i][count])
        zdatadot[i].append(Zr[i][count])

        lines[i][0].set_xdata(np.array(xdata[i]))
        lines[i][0].set_ydata(np.array(ydata[i]))
        lines[i][0].set_3d_properties(np.array(zdata[i]))

        dots[i]._offsets3d = (xdatadot[i], ydatadot[i], zdatadot[i])
    return lines, dots, axes_all, axes_allr
    
ani = FuncAnimation(fig, update, frames=range(0, len(xt[0])), interval=1, repeat=False)
plt.show()

