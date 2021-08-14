#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\H.A.R\Downloads\1.jpg",1)
heightImg = 640
widthImg  = 480
img = cv2.resize(img,(widthImg,heightImg))

img1 = img.copy()
img2 = img.copy()

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edged = cv2.Canny(gray, 70, 200)

kernal = np.ones((5,5))
dilate = cv2.dilate(edged, kernal, iterations = 2)
erode = cv2.erode(dilate, kernal, iterations = 1)

contours, hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

biggest = np.array([])
max_area = 0
for i in contours:
    area = cv2.contourArea(i)
    if area > 5000:
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        if area > max_area and len(approx) == 4:
            biggest = approx
            max_area = area

cv2.drawContours(img1, contours, -1, (0,255,0),10)
cv2.drawContours(img2, [biggest], -1, (0,255,0), 2)
print(biggest)

pts1 = np.float32([biggest[0], biggest[3], biggest[1], biggest[2]])
pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
M = cv2.getPerspectiveTransform(pts1,pts2)
pers = cv2.warpPerspective(img2, M, (widthImg, heightImg))

plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(img2[...,::-1])
plt.subplot(122);plt.imshow(pers[...,::-1])


# In[ ]:




