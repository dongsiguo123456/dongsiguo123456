#!/usr/bin/env python
# coding: utf-8

# In[2]:




import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util
PATH_TO_CKPT = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_LABELS = 'ssd_mobilenet_v1_coco_2017_11_17/mscoco_label_map.pbtxt'
#代表有90中类别
NUM_CLASSES = 90
detection_graph = tf.Graph()
#加载模型
with detection_graph.as_default():
	od_graph_def = tf.compat.v1.GraphDef()
	with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		od_graph_def.ParseFromString(fid.read())
		tf.import_graph_def(od_graph_def, name='')
#load_labelmap代表指定路径
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
#将文件中的类别名字提取出来，和id
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#把图片转换为numpy数组
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    #reshape代表改变数组的顺序
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#导入图片
TEST_IMAGE_PATHS = ['cat.jpg', 'img_1.png']


#将图片导入模型进行训练
#detection_graph为载入的模型，graph=detection_graph将模型导入
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        #image_tensor为输入的函数
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        #剩下四个为输出的结果
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            image_np = load_image_into_numpy_array(image)
            image_np=cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
            #image_np_expanded 扩展一维，现在为四维数组的tenso,第一维代表有几张图片
            image_np_expanded = np.expand_dims(image_np, axis=0)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),
                                                               np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                                                               category_index, use_normalized_coordinates=True,
                                                               line_thickness=8)
            cv2.imshow('1',cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




