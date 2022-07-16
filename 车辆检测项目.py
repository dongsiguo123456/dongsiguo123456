#!/usr/bin/env python
# coding: utf-8

# In[1]:


#形态学识别车辆
import numpy as np
import cv2 

# 1.获取视频对象
cap = cv2.VideoCapture('./test.avi')
mog=cv2.createBackgroundSubtractorMOG2()
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
min_w=35
min_h=70
# 2.判断是否读取成功
while(cap.isOpened()):
    # 3.获取每一帧图像
    ret, frame = cap.read()
    # 4. 获取成功显示图像
    if ret == True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #去除一些噪声污染，需要先使用高斯滤波，在进行开闭运算
        blur=cv2.GaussianBlur(gray,(3,3),5)
        fgmask=mog.apply(blur)
        #查找轮廓
        contours,hierarchy= cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #画出所有的轮廓
        for contour in contours:
             #最大的外接矩形
            x,y,w,h=cv2.boundingRect(contour)
            is_valid=(w>=min_w)&(h>=min_h)
            if not  is_valid:
                continue
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imshow('2',frame)
    key=cv2.waitKey(25)
    # 5.每一帧间隔为25ms
    if key==27:
        break
# 6.释放视频对象
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[2]:





# In[ ]:





# In[ ]:





# In[ ]:




