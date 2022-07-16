# 几篇博客帮助理解

1.向前传播forward函数：[](https://blog.csdn.net/u011501388/article/details/84062483)

2.如何理解卷积神经网络中的通道：[](https://blog.csdn.net/weixin_38481963/article/details/109920512)

3.卷积核和过滤器：[](https://blog.csdn.net/weixin_38481963/article/details/109906338)

![img](https://img-blog.csdn.net/20160707204048899)

4.卷积神经网络：[](https://blog.csdn.net/v_JULY_v/article/details/51812459)

`AttributeError: 'NoneType' object has no attribute 'copy'`

考虑：

问题在代码一开始的问题出现了 说明在我代码cv2.imread（）中读取图像失败，得到的是None类型的变量

解决：

①检查图片路径是否写对

②检查图片命名或路径中是否有中文

### 最大值和最小值及它们的位置

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(imgray,mask = mask)

在卷积层中，每一层的卷积核是不一样的。比如AlexNet

第一层：96*11*11（96表示卷积核个数，11表示卷积核矩阵宽*高） stride（步长） = 4 pad（边界补零） = 0

第二层：256*5*5 stride（步长） = 1 pad（边界补零） = 2

第三，四层：384*3*3 stride（步长） = 1 pad（边界补零） = 1

第五层：256*3*3 stride（步长） = 1 pad（边界补零） = 2

先看sigmoid，tanh分布

 sigmoid函数的功能是相当于把一个实数压缩至0到1之间。当z是非常大的正数时，g(z)会趋近于1，而z是非常小的负数时，则g(z)会趋近于0

![image-20220710174751696](C:\Users\dsg\AppData\Roaming\Typora\typora-user-images\image-20220710174751696.png)

### 2.4 全连接（full connection）

作用：分类器角色，将特征映射到样本标记空间，本质是矩阵变换（affine）。

### 2.6 前向传播（forward propagation）

前向传播包含之前的卷积，Relu激活函数，池化（pool），全连接(fc)，可以说，在损失函数之前操作都属于前向传播。

主要是权重参数w , b 初始化，迭代，以及更新w, b,生成分类器模型。

### 2.7 反向传播（back propagation）

反向传播包含损失函数，通过梯度计算dw，db，Relu激活函数逆变换，反池化，反全连接。

### 2.8 随机梯度下降（sgd_momentum）

作用：由梯度grad计算新的权重矩阵w

![img](https://img-blog.csdn.net/20160702205047459)



