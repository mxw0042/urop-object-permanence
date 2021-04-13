import csv
import numpy as np
from spatialmath import *
from spatialmath.pose3d import * 
from math import pi

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

filename = "tf.csv"

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
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(0, 1)

quiver_10 = ax.quiver(0, 0, 0, 0, 0, 0)
quiver_7 = ax.quiver(0, 0, 0, 0, 0, 0)
quiver_3 = ax.quiver(0, 0, 0, 0, 0, 0)
quiver_4 = ax.quiver(0, 0, 0, 0, 0, 0)
quiver_12 = ax.quiver(0, 0, 0, 0, 0, 0)

def update(count):
    global quiver_10, quiver_7, quiver_3, quiver_4, quiver_12
    row=rows[count]
    print(count)
    if '"obj-10"' in row:
        quiver_10.remove()   
        q=UnitQuaternion([float(row[18]), float(row[15]),float(row[16]),float(row[17])])
        quiver_10 = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q.v[0], q.v[1], q.v[2])
    elif '"obj-7"' in row:
        quiver_7.remove()
        q=UnitQuaternion([float(row[18]), float(row[15]),float(row[16]),float(row[17])])
        quiver_7 = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q.v[0], q.v[1], q.v[2])
    elif '"obj-3"' in row:
        quiver_3.remove()
        q=UnitQuaternion([float(row[18]), float(row[15]),float(row[16]),float(row[17])])
        quiver_3 = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q.v[0], q.v[1], q.v[2])
    elif '"obj-4"' in row:
        quiver_4.remove()
        q=UnitQuaternion([float(row[18]), float(row[15]),float(row[16]),float(row[17])])
        quiver_4 = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q.v[0], q.v[1], q.v[2])
    elif '"obj-12"' in row:
        quiver_12.remove()
        q=UnitQuaternion([float(row[18]), float(row[15]),float(row[16]),float(row[17])])
        quiver_12 = ax.quiver(float(row[11]), float(row[12]), float(row[13]), q.v[0], q.v[1], q.v[2])

ani = FuncAnimation(fig, update, frames=range(1500, len(rows)), interval=10, repeat=False)
plt.show()
