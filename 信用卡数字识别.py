#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
#对轮廓列表进行排序
def sort_contours(cnts,method="left-to-right"):
    reverse =False
    i=0
    if method=="left-to-right"or method=="bottom-to-top":
        reverse =True
    if method=="top-to-bottom"or method=="bottom-to-top":
        i=1
    #利用最小的矩形拟合轮廓
    boundingBoxes=[cv2.boundingRect(c)for c in cnts]
    #利用zip降轮廓信息和对应的矩形坐标组成元祖
    #降元祖排序，根据横坐标从小到大排序轮廓，在对每个轮廓按照坐标从小到大排序
    #最后利用zip(*)将排序的元祖返回轮廓信息和矩阵坐标
    (cnts,boundingBoxes)=zip(*sorted(zip(cnts,boundingBoxes),
                                     key = lambda b: b[1][i], reverse = reverse))
    return cnts,boundingBoxes
# 图片尺寸变换
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
img = cv2.imread("5.png")
# draw会改变原图，这里做一个备份
img_copy = img.copy()
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(ref, 128, 255, cv2.THRESH_BINARY)
# 这个函数只支持接收单颜色通道图像，否则报错
#cv2.CHAIN_APPROX_SIMPLE 只保留终点坐标
refcnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
#  (0,0,255)代表颜色，2代表粗细  -1 代表展示所有轮廓
cv2.drawContours(img_copy,refcnts,-1,(0,255,0),1)#  (0,0,255)代表颜色，2代表粗细
cv2.imshow('2',img_copy)
cv2.imshow('1',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 2.排序

# In[2]:


refCnts=sort_contours(refcnts,method="left-to-right")[0]


# # 3.分割模块
# 这一步主要将模板图像的10个数字轮廓提取出来变成10个对应模板，方便后面的匹配

# In[3]:


digits={}
for (i, c) in enumerate(refCnts):
	# 计算外接矩形并且resize成合适大小
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# 每一个数字对应每一个模板
	digits[i] = roi


# # 加载信用卡图片

# In[4]:


image = cv2.imread('./credit_card_02.png')
image = resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('3',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 初始化卷积核
# 在形态学操作中，包括腐蚀，膨胀，开运算，闭运算，礼帽，黑帽等运算
# retval=getStructuringElement(shape,ksize[,anchor])

# In[5]:


rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))


# # 形态学操作

# In[6]:


#礼帽操作获取比原始图像更亮的边缘信息
tophat=cv2.morphologyEx(gray,cv2.MORPH_TOPHAT,rectKernel)
cv2.imshow('5',tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()
#ksize=-1相当于用3*3的
#使用soble算子求图像梯度
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")
cv2.imshow('4',gradX)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 通过闭操作（先膨胀，再腐蚀）将数字连在一起

# In[7]:


gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv2.imshow('6',gradX)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


#otsu方法会找到最佳的阈值，需把阈值参数设置为0，cv2.threshold 就会自动寻找最优阈值，并将阈值返回
thresh = cv2.threshold(gradX, 0, 255,
cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
#再来一个闭操作
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel) 
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 轮廓提取
# 调用cv2.findContours进行轮廓提取

# In[9]:


threshCnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = threshCnts
cur_img = image.copy()
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv2.imshow('img',cur_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 提取出数字轮廓

# In[10]:


locs = []
for (i, c) in enumerate(cnts):
	# 计算矩形
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)

	# 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
	if ar > 2.5 and ar < 4.0:

		if (w > 40 and w < 55) and (h > 10 and h < 20):
			#符合的留下来
			locs.append((x, y, w, h))


# # 模板匹配

# In[13]:


output = []
# 遍历每一个轮廓中的数字
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []
    # 根据坐标提取每一个组
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    # 根据坐标提取每一个组
    # 预处理
    group = cv2.threshold(group, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv2.imshow('group',group)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #从大轮廓提取出小轮廓
    digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = sort_contours(digitCnts,method="left-to-right")[0]
    for c in digitCnts:
        # 找到当前数值的轮廓，resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv2.imshow('roi',roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 计算匹配得分
        scores = []
        # 在模板中计算每一个得分,得到的最小的就是我们所需要的模板
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI,cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
            # 得到最合适的数字
            groupOutput.append(str(np.argmax(scores)))
    cv2.rectangle(image, (gX - 5, gY - 5),(gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 1)
    output.extend(groupOutput)


# In[14]:


# 打印结果
cv2.imshow("7", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




