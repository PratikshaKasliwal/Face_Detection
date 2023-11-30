#Face detection is usually performed by classifier
#Classifier is an algorithm that deicides whether a given image is positive or negative , whete a face is present or not
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

#haarcascadeis really sensiive to noise and can give wrong result
#hence they are less used for big project
img=cv.imread('photos/group.jpeg')

cv.imshow('group',img)
#it uses edges to determine whether it is face or not , it doesn't do by colors

gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv2.imshow('Gray',gray)

haar_cascade=cv.CascadeClassifier('haar_face.xml')
faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)#min value removes more noise
#will take an image as an i/p with the mentioned variable and return rectangular coordinates of the face as a list to face on the score

print(f'NUMBER of faces found:= {len(faces_rect)}')

for(x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+h,y+h),(0,255,0), thickness=2)
cv.imshow('Detected Faces',img)
cv.waitKey(0)