import cv2
import numpy as np
#https://github.com/opencv/opencv/tree/master/data/haarcascades 
#clasificador repositorio de detector de rostros 


faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
image = cv2.imread('foto2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


#clasificador

faces = faceClassif.detectMultiScale(gray,
	scaleFactor=1.2,
	minNeighbors=5,
	minSize=(50,50),
	maxSize=(250,250))

for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(255,20,147),3)

cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()