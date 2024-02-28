# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 22:22:15 2024

@author: cengh
"""
import cv2
import time 

cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(width, height)
    