#!/usr/bin/env python
# coding: utf-8

# In[2]:


#导入模块
import cv2 as cv
#打开摄像头
cap = cv.VideoCapture(0)
#采集的照片的数目
num = 1
#录入人员的姓名
face_id = input('录入人员姓名:\n')
#录入人员的标签，每个人的标签不能相同
face_idnum = input('录入人员编号:\n')
print('\n 正在打开摄像头。。。。。。。')
while (cap.isOpened()):
    #捕获摄像头图像
    ret_flag,vshow = cap.read()
    #显示捕获的照片
    cv.imshow("capture_test",vshow)
    #图像刷新的频率，图像才能正常显示出来，返回1ms内按键按下的ASII码。
    k = cv.waitKey(1)&0xff
    #设置按键保存照片
    if k == 27:
        #保存图片（保存路径以及文件命名方式）
        cv.imwrite("C:/Users/dsg/Pictures/Saved Pictures/imgdata/"+str(face_idnum)+"."+str(face_id)+".jpg",vshow)
        print("成功保存第"+str(num)+'张照片'+".jpg")
        print("***********************")
        num += 1
    elif k == ord(' '):
        break
#关闭摄像
cap.release()
#释放图像显示窗
cv.destroyAllWindows()


# In[9]:


import os
import cv2 as cv
from PIL import Image
import numpy as np
 
def getImageAndlabels(path):
    #人脸数据数据
    facesSamples = []
    #人标签
    ids = []
    #读取所有的照片的名称（os.listdir读取根目录下文件的名称返回一个列表，os.path.join将根目录和文件名称组合形成完整的文件路径）
    imagePaths = [os.path.join(path,f) for  f in os.listdir(path)]
    #调用人脸分类器（注意自己文件保存的路径，英文名）
    face_detect = cv.CascadeClassifier('C:/Users/dsg/Downloads/opencv-4.6.0/opencv-4.6.0/data/haarcascades/haarcascade_frontalface_alt2.xml')
    #循环读取照片人脸数据
    for imagePath in imagePaths:
        #用灰度的方式打开照片
        PIL_img = Image.open(imagePath).convert('L')
        #将照片转换为计算机能识别的数组OpenCV（BGR--0-255）
        img_numpy = np.array(PIL_img,'uint8')
        #提取图像中人脸的特征值
        faces = face_detect.detectMultiScale(img_numpy)
        #将文件名按“.”进行分割
        id = int(os.path.split(imagePath)[1].split('.')[0])
        #防止无人脸图像
        for x,y,w,h in faces:
            ids.append(id)
            facesSamples.append(img_numpy[y:y+h,x:x+w])
    return facesSamples,ids
 
 
if __name__ == '__main__':
    #人脸图片存放的文件夹
    path ='imgdata'
    faces, ids = getImageAndlabels(path)
    #调用LBPH算法对人脸数据进行处理
    recognizer = cv.face.LBPHFaceRecognizer_create()
    #训练数据
    recognizer.train(faces, np.array(ids))
    #将训练的系统保存在特定文件夹
    recognizer.write('trainer/trainer.yml')


# In[1]:


import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pyttsx3
engine = pyttsx3.init()
#加载训练数据集文件
recogizer=cv2.face.LBPHFaceRecognizer_create()
#读取训练好的系统文件
recogizer.read('trainer/trainer.yml')
#存储人脸库中人员的名字
names=[]
#对应的标签
idn = []
#准备识别的图片
def face_detect_demo(img):
    #转换为灰度
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #调用下载好的人脸分类器
    face_detector = cv2.CascadeClassifier('C:/Users/dsg/Downloads/opencv-4.6.0/opencv-4.6.0/data/haarcascades/haarcascade_frontalface_alt2.xml')
    #读取图像中人脸的特征值（返回值为人脸的相关坐标和长宽）
    face=face_detector.detectMultiScale(gray,1.1,5,0,(100,100),(800,800))
    for x,y,w,h in face:
        #把人脸框起来
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        # 人脸识别
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        if confidence > 60:
            img = cv2AddChineseText(img, "外来人员"+str(int(confidence)), (x+10, y+10), (0, 255, 0), 30)
            engine.say('陌生人员靠近！')
            engine.runAndWait()
          #putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            img = cv2AddChineseText(img, str(names[idn.index(ids)])+str(int(confidence)), (x+10, y-25), (0, 255, 0), 30)
          #putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
            engine.say(str(names[idn.index(ids)]) + '同学，你好!')
            engine.runAndWait()
    cv2.imshow('result',img)
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB
                                           ))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
def name():
    path = './imgdata/'
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',3)[1])
       id = int(os.path.split(imagePath)[1].split('.',3)[0])
       names.append(name)
       idn.append(id)
cap=cv2.VideoCapture(0)
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()


# In[ ]:




