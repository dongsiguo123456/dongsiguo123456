#!/usr/bin/env python
# coding: utf-8

# In[64]:


import urllib.request
import json
import base64
class BaiduFace(object):
    def __init__(self,app_id,api_key,secret_key):
        self.app_id=app_id
        #将三个参数保存          
        self.api_key=api_key
        self.secret_key=secret_key
        self.token=None
        self.time_exp=None
        self.error=0
        self.err_msg=' '
    def _get_token(self):
        self.error=0

        #获取token
        #判断是否有有效token
                    
        self.err_msg=' '
        if self.token==None:
            
            url = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id='+self.api_key+'&client_secret='+self.secret_key                     
            #url
            #构建url请求
            request=urllib.request.Request(url)
            request.add_header('Content-Type','application/json;charset=UTF-8')
            #发送请求
            response=urllib.request.urlopen(request)
            #解析返回，去除token
            content=response.read()
            content_decoded=json.loads(content.decode())
            if 'access_token'not in content_decoded.keys():
                self.error=1           
                self.err_msg=content_decoded['error']
            self.token=content_decoded['access_token']
            print(self.token)
        return self.error



    def add_user(self,img64,groupid,userid,userinfo=None):
        #添加人脸图片到百度人脸库中
                    
        #param img64:经过base64编码的照片
        #param groupid:用户组id
        #param userid:用户id
        self.error=0
        self.err_msg=''

        #获取token
        if self._get_token()!=0:
             print(self.err_msg)
       
             return self.error
        #填写参数
        params={}
        params['image']=img64
        params['image_type']='BASE64'
        params['group_id']=groupid
        params['user_id']=userid
        if userinfo:
            params['user_info']=userinfo
        params['quality_control']='NORMAL'    
        params['quality_control']='NORMAL'
        params['liveness_control']='LOW'
        params=json.dumps(params).encode()
        


        #发送http请求
        print(self.token)
        url='https://aip.baidubce.com/rest/2.0/face/v3/faceset/user/add'+'?access_token='+self.token
        request=urllib.request.Request(url,params)
        request.add_header('Content-Type','application/json;charset=UTF-8')
        response=urllib.request.urlopen(request)

        #分析结果
        content=response.read()
        content_decoded=json.loads(content.decode())
        if 'error_code'in content_decoded.keys():
            self.error=content_decoded['error_code']
                        
            self_err_msg=content_decoded['error_msg']
        return self.error

bf= BaiduFace('26813838','0BWAuUmiSL0q46ueBeFMVfEP','7yWirgUwvp1uG2FItqq0HOTodyymLZUM')
if __name__=='__main__':
    ret=bf._get_token()
    print(bf.token)
    with open('1.dong.jpg','rb')as f:
        img=f.read()
                    
        img64=base64.b64encode(img).decode()
        ret=bf.add_user(img64,'w','u')
        print(ret)

 
# In[10]:


from PIL import Image
import matplotlib.pyplot as plt
import base64
pic=Image.open('1.jpg')
plt.imshow(pic)
plt.axis('off')
#获取图片：打开文件，获取指定的图像
#1.调用tkinter弹出窗口，进行文件选择。得到文件路径
#2.调用PIL的image类的open()打开图像，show()显示图像
#3.如果在程序窗口显示，则调用matplotlib.pyplot的imshow()显示图像
#4.调用base64的b64encode()将图像转化为base64编码
f=open('1.jpg','rb')
img=base64.b64encode(f.read()).decode('utf-8')
#将人脸检测封装成函数：
#1.确定UPL
#2.确定URL参数+access_token
#3.设置请求参数
#4.调用requests的post()方法传送参数获取检测内容
#5.将检测内容转换为json
#6.从json格式中提取相应的检测信息并输出


# In[ ]:




