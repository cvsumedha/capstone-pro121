import cv2
import time
import numpy as np

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
img = cv2.imread('C:/Users/Chinky/Downloads/68.jpg')


#Starting the webcam
cap = cv2.VideoCapture(0)

#Allowing the webcam to start by making the code sleep for 2 seconds
time.sleep(2)
frame = 0


#Capturing background for 60 frames
for i in range(60):
    ret, bg = cap.read()
#Flipping the background
frame = np.flip(bg, axis=1)

#Reading the captured frame until the camera is open
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    #Flipping the image for consistency
    img = np.flip(img, axis=1)

   
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    u_black = np.array([104, 153, 70])
    l_black = np.array([30,30,0])
    mask = cv2.inRange(rgb, u_black, l_black)
   
    
    res = cv2.bitwise_and(frame, frame, mask= mask)
    
    f = frame - res
    f = np.where(f == 0, img ,f)

    output_file.write(f)
    
    cv2.imshow("magic", f)
    cv2.waitKey(1)



cap.release()
out.release()
cv2.destroyAllWindows()