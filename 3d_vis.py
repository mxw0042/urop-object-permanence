import csv
import numpy as np
from spatialmath import *
from math import pi

import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

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
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for row in rows[:100]:
    count+=1
    if '"obj-10"' in row:   # start with just visualizing one object
        q=UnitQuaternion([float(row[15]),float(row[16]),float(row[17]),float(row[18])])
        ax.scatter(float(row[11]), float(row[12]), float(row[13]))
        p=SE3(float(row[11]), float(row[12]), float(row[13]))
        T=p * q.SO3()
        T.plot(frame='1', dims=[0.1391,0.1395,-0.00025,0.0000,0.8505,0.8530])
        plt.pause(0.0001)
        print(count)
        pose.append(T)
print("done")
plt.show()



# tranimate(transl(4, 3, 4)@trotx(2)@troty(-2), frame='2', arrow=False, dims=[0, 5], nframes=200)
