from tkinter import *
import time
import cv2  
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import ImageGrab

#Window class for making different windows
class Window:

    #Constructor
    def __init__(self, window, width=600, height=400):
        #Set variables
        self.width = width
        self.height = height

        #This dictionary is used to keep track of an item being dragged
        self._drag_data = {"x": 0, "y": 0, "item": None}

        #Create canvas
        self.canvas = Canvas(window, height=self.height, width=self.width)
        self.canvas.pack()

        button1 = Button(self.canvas, text = "cover", command = self.cover, anchor = W)
        button1_window = self.canvas.create_window(30, 30, anchor=S, window=button1)
        button2 = Button(self.canvas, text = "uncover", command = self.uncover, anchor = W)
        button2_window = self.canvas.create_window(30, 30, anchor=N, window=button2)

        #Add bindings for clicking, dragging and releasing over any object with the "circledrag" tag
        # self.canvas.bind('<B1-Motion>', move)
        self.canvas.tag_bind("circledrag", "<ButtonPress-1>", self.OnCircleButtonPress)
        self.canvas.tag_bind("circledrag", "<ButtonRelease-1>", self.OnCircleButtonRelease)
        self.canvas.tag_bind("circledrag", "<B1-Motion>", self.OnCircleMotion)

    #This is used to draw particle objects on the canvas, notice the tag that has been added as an attribute
    def _create_circle(self, xcoord, ycoord, color, tag):
        self.canvas.create_oval(xcoord-25, ycoord-25, xcoord+25, ycoord+25, outline=color, fill=color, tags = ("circledrag", "circle", tag))

    def _create_cup(self, xcoord, ycoord):
        global img
        img = PhotoImage(file="C:/Users/mxw00/Pictures/red_cup.png")
        self.canvas.create_image(xcoord, ycoord, image=img, tags=("cup", "circledrag"))

    def cover(self):
        cup = self.canvas.find_withtag('cup') 
        corners=self.canvas.bbox(cup[0])
        overlap=self.canvas.find_enclosed(corners[0], corners[1], corners[2], corners[3])
        tag_list = self.canvas.gettags(overlap[0])
        for tag in tag_list:
            if tag.startswith("circle-"):
                print(tag)
                self.canvas.addtag_withtag(tag, "cup")

    def uncover(self):
        cup = self.canvas.find_withtag('cup') 
        corners=self.canvas.bbox(cup[0])
        overlap=self.canvas.find_enclosed(corners[0], corners[1], corners[2], corners[3])
        tag_list = self.canvas.gettags(overlap[0])
        for tag in tag_list:
            if tag.startswith("circle-"):
                print(tag)
                self.canvas.dtag("cup", tag)
        

    #This uses the find_closest method to get store the x and y positions of the nearest item into the dictionary
    def OnCircleButtonPress(self, event):
        #print self.canvas.find_withtag("Current")

        '''Begin drag of an object'''
        # record the item and its location
        item = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags(item)
        for tag in tags:
            if tag.startswith("circle-"):
                break
        self._drag_data["item"] = tag
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    #This clears the dictionary once the mouse button has been released
    def OnCircleButtonRelease(self, event):
        '''End drag of an object'''
        # reset the drag information
        self._drag_data["item"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    #This moves the item as it is being dragged around the screen
    def OnCircleMotion(self, event):
        '''Handle dragging of an object'''
        # compute how much this object has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        # move the object the appropriate amount
        self.canvas.move(self._drag_data["item"], delta_x, delta_y)
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

class DragCircle:
#Constructor
    def __init__(self, window, width=100, height=100, colour="red"):
        self.window = window
        tag = "circle-%d" % id(self)
        self.circle = self.window._create_circle(width, height, colour, tag)

class DragCup:
#Constructor
    def __init__(self, window, width=100, height=100):
        self.window = window
        tag = "cup-%d" % id(self)
        self.cup = self.window._create_cup(width, height)
        


# Threshold of green in HSV space 
lower_green = np.array([50, 100, 100]) 
upper_green = np.array([70, 255, 255]) 

# Threshold of blue in HSV space 
lower_blue = np.array([110,200,200])
upper_blue = np.array([130,255,255])

# Threshold of yellow in HSV space 
lower_yellow = np.array([30, 90, 90]) 
upper_yellow = np.array([40, 255, 255])

#Make a window from my own class
window = Tk()
window.title("Drag & Drop")

#Create an instance of the window class
main_window = Window(window)

#Create a circle object from the DragCircle class
circle = DragCircle(main_window, 100, 100, "yellow")
circle2 = DragCircle(main_window, 200, 200, "green")
circle3 = DragCircle(main_window, 300, 300, "blue")
cup= DragCup(main_window, 50, 200)

#Start the animation loop
def task():
    x=window.winfo_rootx()+main_window.canvas.winfo_x()
    y=window.winfo_rooty()+main_window.canvas.winfo_y()
    x1=x+main_window.canvas.winfo_width()+170
    y1=y+main_window.canvas.winfo_height()+150
    
    img=ImageGrab.grab((x,y,x1,y1)).save("test.png")
    frame = cv2.imread("test.png")
    # It converts the BGR color space of image to HSV color space 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

    # preparing the mask to overlay 
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
    mask_green = cv2.inRange(hsv, lower_green, upper_green) 
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) 
        
    # The black region in the mask has the value of 0, 
    # so when multiplied with original image removes all non-blue regions 
    result = cv2.bitwise_or(mask_blue, mask_green) 
    result = cv2.bitwise_or(result, mask_yellow) 
    # cv2.imshow('frame', frame) 
    # cv2.imshow('mask', mask_blue) 
    # cv2.imshow('mask', mask_green) 
    # cv2.imshow('mask', mask_yellow) 
    cv2.imshow('result', result) 
    window.after(500, task)  # reschedule event in 2 seconds
    

window.after(500, task)
window.mainloop()

# Closes all the frames 
cv2.destroyAllWindows() 