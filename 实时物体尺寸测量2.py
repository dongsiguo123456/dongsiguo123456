import cv2
import numpy as np
#以在一个A4纸为一个背景上的图像的物体尺寸测量
#大致思路：第一步，进行轮廓提取，将图片转化为灰度图，然后进行高斯模糊，模糊后提取轮廓，然后进行膨胀收缩使轮廓更加的清晰
# 然后找出最大轮廓，写一个函数：得出的轮廓四角的顺序可能发生变化，因此需要将四角的坐标固定为矩形一种固定的顺序
#将最大矩形的A4纸透视还原为正常的长方形，然后找出A4纸中的图形轮廓，计算实际测量的大小

img = cv2.imread('img_6.png')
img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
minArea = 1000
filter = 4
scale =2
wp=210*scale
hp =297*scale

#提取轮廓的函数，对轮廓进行形态学的操作如滤波，膨胀等使轮廓更加清晰
def getContours(img):
    imgG = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #高斯滤波操作
    imgBlur = cv2.GaussianBlur(imgG,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,100,100)
    #初始化核结构
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    cv2.imshow('res',imgCanny)
    cv2.imshow('res2',imgThre)

    contours, hiearchy = cv2.findContours(imgThre,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area> minArea:
            # 计算轮廓的周长，true表示轮廓为封闭
            peri = cv2.arcLength(i,True)
            #进行轮廓的多边拟合
            appprox = cv2.approxPolyDP(i,0.02*peri,True)
            bbox = cv2.boundingRect(appprox)
            if filter > 0 :
                if(len(appprox))==filter:
                    finalCountours.append([len(appprox),area,appprox,bbox,i])
            else:
                finalCountours.append([len(appprox), area, appprox, bbox, i])
    # 对第二个数值面积进行排序，为升序，找出轮廓的最大值
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    for con in finalCountours:
        cv2.drawContours(img,con[4],-1, (0, 0, 255), 4)
    return img,finalCountours


#重新排序函数：对轮廓坐标以一种固定的顺序排序
def reorder(myPoints):
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    myPointsNew[1]= myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    print(myPoints)
    print(myPoints)
    return myPointsNew

#将最大矩形的A4纸透视还原为正常的长方形
def warpImg (img,points,w,h,pad=20):
    # print(points)
    points =reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp
#计算图像的尺寸的函数
def findDis(pts1,pts2):
    return ((pts2[0]-pts1[0])**2 + (pts2[1]-pts1[1])**2)**0.5

#找到轮廓中图像的轮廓
imgcon, cons= getContours(img)
if(len(cons)!=0):
    maxbox = cons[0][2]
    reorder(maxbox)
    cv2.imshow('img', imgcon)
    #重新排序函数
    imgWarp = warpImg(imgcon, maxbox, wp, hp)
    cv2.imshow('imgWarp',imgWarp)
    imgcon2, cons2 = getContours(imgWarp)
    if(len(cons2)!=0):
        for obj in cons2:
            cv2.polylines(imgcon2,[obj[2]],True,(0,255,0),2)
            nPoints = reorder(obj[2])
            nW = round((findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
            nH = round((findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)
            cv2.arrowedLine(imgcon2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgcon2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                            (255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = obj[3]
            cv2.putText(imgcon2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)
            cv2.putText(imgcon2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                        (255, 0, 255), 2)

    cv2.imshow('img2',imgcon2)

cv2.waitKey(0)


