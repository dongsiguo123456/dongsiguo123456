#!/usr/bin/env python
# coding: utf-8

# In[42]:


import dlib
from imageio import imread
import glob
import numpy as np


# In[40]:


detector = dlib.get_frontal_face_detector()
#建立一个窗口
win = dlib.image_window()
paths = glob.glob('hao.jpg')
for path in paths:
	img = imread(path)
	# 1 表示将图片放大一倍，便于检测到更多人脸
	dets = detector(img, 1)
	print('检测到了 %d 个人脸' % len(dets))
	for i, d in enumerate(dets):
		print('- %d：Left %d Top %d Right %d Bottom %d' % (i, d.left(), d.top(), d.right(), d.bottom()))
 #清空
	win.clear_overlay()
	win.set_image(img)
	win.add_overlay(dets)


# In[3]:


#设置一个阈值
path = 'six.jpeg'
img = imread(path)
# -1 表示人脸检测的判定阈值
# scores 为每个检测结果的得分，idx 为人脸检测器的类型
#detector.run()为判断函数
#-1为设置的阈值
dets, scores, idx = detector.run(img, 1, -1)
for i, d in enumerate(dets):
	print('%d：score %f, face_type %f' % (i, scores[i], idx[i]))
win.clear_overlay()
win.set_image(img)
win.add_overlay(dets)


# # 人脸的关键点检测

# In[16]:


detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
win = dlib.image_window()
paths = glob.glob('six.jpeg')
for path in paths:
	img = imread(path)
	win.clear_overlay()
	win.set_image(img)
 
	# 1 表示将图片放大一倍，便于检测到更多人脸
	dets = detector(img, 1)
	print('检测到了 %d 个人脸' % len(dets))
	for i, d in enumerate(dets):
		print('- %d: Left %d Top %d Right %d Bottom %d' % (i, d.left(), d.top(), d.right(), d.bottom()))
		shape = predictor(img, d)
		# 第 0 个点和第 1 个点的坐标
		print('Part 0: {}, Part 1: {}'.format(shape.part(0), shape.part(1)))
		win.add_overlay(shape)
 
	win.add_overlay(dets)
	dlib.hit_enter_to_continue()


# # 人脸识别

# In[44]:


detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat'
#facere得到了一个人脸检测的模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
labeled = glob.glob('lena.jpg')
labeled_data = {}
unlabeled = glob.glob('hao.jpg')
# 定义一个计算Euclidean距离的函数
def distance(a, b):
	# d = 0
	# for i in range(len(a)):
	# 	d += (a[i] - b[i]) * (a[i] - b[i])
	# return np.sqrt(d)
	return np.linalg.norm(np.array(a) - np.array(b), ord=2)
# 读取标注图片并保存对应的128向量
for path in labeled:
	img = imread(path)
    #rstrip函数返回删除 string 字符串末尾的指定字符后生成的新字符串
	name = path.rstrip('.jpg')
	dets = detector(img, 1)
	# 这里假设每张图只有一个人脸
	shape = predictor(img, dets[0])
	face_vector = facerec.compute_face_descriptor(img, shape)
	labeled_data[name] = face_vector
# 读取未标注图片，并和标注图片进行对比
for path in unlabeled:
	img = imread(path)
	name = path.rstrip('.jpg')
	dets = detector(img, 1)
	# 这里假设每张图只有一个人脸
	shape = predictor(img, dets[0])
	face_vector = facerec.compute_face_descriptor(img, shape)
	matches = []
	for key, value in labeled_data.items():
		d = distance(face_vector, value)
		matches.append(key + ' %.2f' % d)
            
	print('{}：{}'.format(name, ';'.join(matches)))


# In[ ]:





# In[ ]:




