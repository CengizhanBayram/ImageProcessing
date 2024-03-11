# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:22:15 2024

@author: cengh
"""
import cv2
import numpy as np 
from collections import deque

buffer_size = 16
pts = deque(maxlen=buffer_size)
#mavi renk aralığı 
blue_lower =(84, 98, 0)
blue_upper =(179, 255, 255)

cap = cv2.VideoCapture(0)

cap.set(3, 960)
cap.set(4, 480)

while True :
    sucess , imgoriginal = cap.read()
    if(sucess):
        blurred = cv2.GaussianBlur(imgoriginal, (11,11), 0)
        hsv =cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        #mavi için maske oluşturma 
        mask = cv2.inRange(hsv, blue_lower, blue_upper)
        mask = cv2.erode(mask, None, iterations= 2 )
        mask = cv2.dilate(mask, None, iterations= 2 )
        # kontur
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None 
        if (len(contours)>0):
            # en büyük kontoru al 
            c = max(contours , key =cv2.contourArea)
            rect =cv2.minAreaRect(c)
            ((x,y),(width, height),rotation) = rect
            s = "x:{},y:{},width :{},height :{},rotation:{}".format(np.round(x),np.round(y),np.round(height),np.round(width),np.round(rotation) )
            box = cv2.boxPoints(rect)
            box = np.int64(box)
                #moment
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
            # konturu çizdirme 
            cv2.drawContours(imgoriginal, [box], 0, (0,255,255), 2 )
            #merkeze bir tane nokta çizdirmek 
            cv2.circle(imgoriginal ,center, (255,0,255), -1)
            cv2.putText(imgoriginal, s , (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0))
            cv2.imshow("last one", imgoriginal)
    if cv2.waitKey(1) & 0xFF == ord("q"):break
