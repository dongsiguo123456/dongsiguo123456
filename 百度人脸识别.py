#!/usr/bin/env python
# coding: utf-8

# In[ ]:


1.获取token
https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=【应用的AK】&client_secret=【应用的SK】
利用token去网站进行检测
https://aip.baidubce.com/rest/2.0/face/v1
2.准备照片，填写参数
3.调用人脸检测的api
4.显示检测结果


# In[3]:


def getToken():
    ak = '08AZrn9NUgNVprkVOxp6D2Ca'
    sk = 'Nwj1Bfrm2tpsFfSYch9LzGFu4CM6IhIF'
    host = f'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={ak}&client_secret={sk}'
    #创建http请求
    response = requests.get(host)
    return response.json().get("access_token")

def img_to_base64(file_path):
    with open(file_path, 'rb') as f:
        base_64_data = base64.b64encode(f.read())
        s = base_64_data.decode()
        return s

def FaceDetect(token_, base_64_data):
    params = {}
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
    params["image"] = base_64_data
    params["image_type"] = "BASE64"
    params["face_field"] = "age,beauty"
    access_token = token_
    request_url = request_url + "?access_token=" + access_token
    headers = {'content-type': 'application/json'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        pprint(response.json())
        print(response.json()["result"]["face_list"][0]["age"])
        print(response.json()["result"]["face_list"][0]["beauty"])
    return response.json()

import requests
import base64
import json
from pprint import pprint
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mping
if __name__ == "__main__":
    img=mping.imread("seven.jpg")
    base_64 =img_to_base64("seven.jpg")
    token  = getToken()
    content=FaceDetect(token, base_64)
    result=content['result']
    print(result['face_num'])
    for face in result['face_list']:
        loc=face['location']
        plt1=(int(loc['left']),int(loc['top']))
        plt2=(int(loc['left'])+int(loc['width']),int(loc['top'])+int(loc['height']))
        cv2.rectangle( img,plt1,plt2,color=(255,0,0),thickness=2)
        plt.figure(figsize=(10,8),dpi=100)
        plt.imshow(img)
        plt.axis('off')
        plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




