import torch
import torchvision
import torch.nn.functional as F
import torch.optim as optim  # 优化器
from model import facenet
from torch.utils.data import DataLoader
from dataset import train_set, val_set
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


batch_size=32
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
#实例化卷积神经网络
network = facenet()
network.train()
#将模型传入cpu
network = network .to(device)
# 查看网络结构
print("查看网络结构")
print(network )
loss_function = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
# 优化器选择Adam，学习率设为0.001
optimizer = optim.SGD(network.parameters(), lr=0.001)
def train():
    for epoch in range(15):  # 整个迭代15轮
        running_loss = 0.0  # 初始化损失函数值loss=0
        for i, data in enumerate(train_loader,
                                 start=0):
            # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串）
            # enumerate将其组成一个索引序列，利用它可以同时获得索引和值
            # enumerate多用于在for循环中得到计数，enumerate 是python自带的一个函数，start为索引起始值
            # 获取训练数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # 将数据及标签传入GPU/CPU

            # 权重参数梯度清零
            optimizer.zero_grad()

            # 正向及反向传播
            outputs = network(inputs)  # 调用上面的神经网络，正向传播
            loss = loss_function(outputs, labels)  # 损失函数计算经过卷积神经网络后的值与原来的值的差距
            loss.backward()  # 调用pytorch的自动反向传播函数，自动生成梯度
            optimizer.step()  # 执行优化器，把梯度传播回每个网络

            value,index=torch.max(outputs,dim=1)
            #correct=(index==labels).sum().item()
            # 显示损失值
            #acc=correct/batch_size
            #running_loss += loss.item()
            # item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。python自带的字典遍历函数
            if i % 25 == 0:
                #print("epoch:%s\tloss:%.6s\tacc:%.4s"%(epoch,loss.item(),acc))
                print("epoch:%s\tloss:%.6s" % (epoch,loss.item()))
                #print("predict:",[i.item() for i in index])
                #print("truth:",[i.item() for i in labels])


        if epoch % 3 == 0:
            print('saving model')
            torch.save(network.state_dict(),'./model.pt')
train()
'''测试模型'''
correct = 0
total = 0
# with是一个上下文管理器
# with torch.no_grad()表示其包括的内容不需要计算梯度，也不会进行反向传播，节省内存
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = network(images)
        # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 100 test images: %d %%' % (100 * correct / total))



