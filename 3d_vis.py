import csv
import numpy as np
from spatialmath import *
from spatialmath.pose3d import * 
from math import pi

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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
  
count=0
fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

# def get_arrow(x, y, z, q):
#     u = np.sin(2*theta)
#     v = np.sin(3*theta)
#     w = np.cos(3*theta)
#     return x,y,z,u,v,w

# ax.set_xlim(0, 1)
# ax.set_ylim(-.5, .5)
# ax.set_zlim(0.5, 1)
ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_zlim(0.5, 1)

axes_10=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]
axes_7=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]
axes_3=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]
axes_4=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]
axes_12=[ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0), ax.quiver(0, 0, 0, 0, 0, 0)]

objs = {
  '"obj-10"': axes_10,
  '"obj-7"': axes_7,
  '"obj-3"': axes_3,
  '"obj-4"': axes_4,
  '"obj-12"': axes_12
}

colors=['#E16F6D', '#4eb604', '#0d87c3']

def update(count):
    global axes_10, axes_7, axes_3, axes_4, axes_12
    row=rows[count]
    print(count)
    q=UnitQuaternion([float(row[18]), float(row[15]),float(row[16]),float(row[17])]).R
    if '"obj-10"' in row:
        for i in range(3):
            axes_10[i].remove()   
            axes_10[i] = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q[0][i], q[1][i], q[2][i], length=0.1, normalize=True, color=colors[i])
    elif '"obj-7"' in row:
        for i in range(3):
            axes_7[i].remove()   
            axes_7[i] = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q[0][i], q[1][i], q[2][i], length=0.1, normalize=True, color=colors[i])
    elif '"obj-3"' in row:
        for i in range(3):
            axes_3[i].remove()   
            axes_3[i] = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q[0][i], q[1][i], q[2][i], length=0.1, normalize=True, color=colors[i])
    elif '"obj-4"' in row:
        for i in range(3):
            axes_4[i].remove()   
            axes_4[i] = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q[0][i], q[1][i], q[2][i], length=0.1, normalize=True, color=colors[i])
    elif '"obj-12"' in row:
        for i in range(3):
            axes_12[i].remove()   
            axes_12[i] = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q[0][i], q[1][i], q[2][i], length=0.1, normalize=True, color=colors[i])

ani = FuncAnimation(fig, update, frames=range(0, len(rows)), interval=10, repeat=False)
plt.show()
