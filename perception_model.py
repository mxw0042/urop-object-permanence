# importing cv2  
import cv2  
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
 

cap = cv2.VideoCapture(0) 

# Check if camera opened successfully 
if (cap.isOpened() == False):  
  print("Error opening video file") 
   
# Threshold of green in HSV space 
lower_green = np.array([50, 100, 100]) 
upper_green = np.array([70, 255, 255]) 

# Threshold of blue in HSV space 
lower_blue = np.array([110,200,200])
upper_blue = np.array([130,255,255])

# Threshold of yellow in HSV space 
lower_yellow = np.array([30, 90, 90]) 
upper_yellow = np.array([40, 255, 255])

# path  
path = r'C:/Users/mxw00/Pictures/cup_game_moved.png'
  
# Reading an image in default mode 
image = cv2.imread(path)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
mask_green = cv2.inRange(hsv, lower_green, upper_green) 
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) 
result = cv2.bitwise_or(mask_blue, mask_green) 
result = cv2.bitwise_or(result, mask_yellow) 
masks=[mask_blue, mask_green, mask_yellow]
colors=["Blue", "Green", "Yellow"]
for i in range(len(masks)):
    M=cv2.moments(masks[i])
    print(M)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(result, colors[i]+":", (cX-10, cY-5),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
    cv2.putText(result,"("+str(cX)+", "+str(cY)+")", (cX-20, cY+5),cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1)
# cv2.imshow('frame', frame) 
# cv2.imshow('mask', mask_blue) 
# cv2.imshow('mask', mask_green) 
# cv2.imshow('mask', mask_yellow) 
cv2.imshow('result', result) 
cv2.waitKey(0)



# while(cap.isOpened()): 
#     ret, frame = cap.read() 
#     if ret == True: 
#         # It converts the BGR color space of image to HSV color space 
#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 

#         # preparing the mask to overlay 
#         mask_blue = cv2.inRange(hsv, lower_blue, upper_blue) 
#         mask_green = cv2.inRange(hsv, lower_green, upper_green) 
#         mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow) 
            
#         # The black region in the mask has the value of 0, 
#         # so when multiplied with original image removes all non-blue regions 
#         result = cv2.bitwise_or(mask_blue, mask_green) 
#         result = cv2.bitwise_or(result, mask_yellow) 
#         # cv2.imshow('frame', frame) 
#         # cv2.imshow('mask', mask_blue) 
#         # cv2.imshow('mask', mask_green) 
#         # cv2.imshow('mask', mask_yellow) 
#         cv2.imshow('result', result) 
#         # Press Q on keyboard to  exit 
#         if cv2.waitKey(25) & 0xFF == ord('q'): 
#             break
#     else:  
#         break
    
# # When everything done, release  
# # the video capture object 
# cap.release() 
   
# Closes all the frames 
cv2.destroyAllWindows() 
