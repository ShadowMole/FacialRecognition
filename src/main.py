import cv2
import numpy
import os

for i in range(4):
    imname = '\\..\\..\\src\\TrainingImages\\Steven'+str(i+1)+'.jpg'
    path = os.getcwd() + imname
    print(path)
    image = cv2.imread(imname,0)
    cv2.imshow('hi',image)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vector = []
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            vector.append(gray[i,j])
