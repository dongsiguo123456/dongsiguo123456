import torch
#卷积神经网络中输出层的上游通常是全连接层
#对于图像分类问题，输出层使用逻辑函数或归一化指数函数（softmax function）输出分类标签 。
#在物体识别（object detection）问题中，输出层可设计为输出物体的中心坐标、大小和分类 。
#在图像语义分割中，输出层直接输出每个像素的分类结果
import torch.nn as nn
import torch.nn.functional as F
from torch import nn,optim
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # 优化器
'''下载数据集'''
# transforms.Compose()函数将两个函数拼接起来。
# （ToTensor()：把一个PIL.Image转换成Tensor，Normalize()：标准化，即减均值，除以标准差）
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 训练集：下载CIFAR10数据集，如果没有事先下载该数据集，则将download参数改为True
trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform)
# 用DataLoader得到生成器，其中shuffle：是否将数据打乱；
# num_workers表示使用多进程加载的进程数，0代表不使用多进程
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

# 测试集数据下载
testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 显示图像
def imshow(img):
    # 因为标准化normalize是：output = (input-0.5)/0.5
    # 则反标准化unnormalize是：input = output*0.5 + 0.5
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # transpose()会更改多维数组的轴的顺序
    # Pytorch中是[channel，height，width],这里改为图像的[height，width，channel]
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机获取部分训练数据
# next() 返回迭代器的下一个项目。
# next() 函数要和生成迭代器的 iter() 函数一起使用。
dataiter = iter(trainloader)
images, labels = dataiter.next()
# 显示图像
# torchvision.utils.make_grid()将多张图片拼接在一张图中
imshow(torchvision.utils.make_grid(images))
# 打印标签
# str.join(sequence)：用于将序列中的元素以指定的字符str连接成一个新的字符串。这里的str是' '，空格
# %5s：表示输出字符串至少5个字符，不够5个的话，左侧用空格补
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 有GPU就用GPU跑，没有就用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层1：输入图像深度=3，输出图像深度=16，卷积核大小=5*5，卷积步长=1;16表示输出维度，也表示卷积核个数
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1)
        # 池化层1：采用最大池化，区域集大小=2*2.池化步长=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 卷积层2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1)
        # 池化层2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层1：输入大小=32*5*5，输出大小=120
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        # 全连接层2
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # input(3, 32, 32) output(16, 28, 28)
        x = self.pool1(x)  # output(16, 14, 14)
        x = F.relu(self.conv2(x))  # output(32, 10, 10)
        x = self.pool2(x)  # output(32, 5, 5)
        x = x.view(-1, 32 * 5 * 5)  # output(32*5*5)
        x = F.relu(self.fc1(x))  # output(120)
        x = F.relu(self.fc2(x))  # output(84)
        x = self.fc3(x)  # output(10)
        return x


net = LeNet()
net = net.to(device)
# 查看网络结构
print("查看网络结构")
print(net)
# 接下来进行模型训练
net = LeNet()
net = net.to(device)
loss_function = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
# 优化器选择Adam，学习率设为0.001
optimizer = optim.Adam(net.parameters(), lr=0.001)
for epoch in range(10):  # 整个迭代10轮
    running_loss = 0.0  # 初始化损失函数值loss=0
    for i, data in enumerate(trainloader,
                             start=0):  # 对于一个可迭代的（iterable）/可遍历的对象（如列表、字符串），enumerate将其组成一个索引序列，利用它可以同时获得索引和值
        # enumerate多用于在for循环中得到计数，enumerate 是python自带的一个函数，start为索引起始值
        # 获取训练数据
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据及标签传入GPU/CPU

        # 权重参数梯度清零
        optimizer.zero_grad()

        # 正向及反向传播
        outputs = net(inputs)  # 调用上面的神经网络，正向传播
        loss = loss_function(outputs, labels)  # 损失函数计算经过卷积神经网络后的值与原来的值的差距
        loss.backward()  # 调用pytorch的自动反向传播函数，自动生成梯度
        optimizer.step()  # 执行优化器，把梯度传播回每个网络

        # 显示损失值
        running_loss += loss.item()  # item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。python自带的字典遍历函数
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    print('Finished Training')
'''测试模型'''
correct = 0
total = 0
# with是一个上下文管理器
# with torch.no_grad()表示其包括的内容不需要计算梯度，也不会进行反向传播，节省内存
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# 打印10个分类的准确率
class_correct = list(0. for i in range(10))  # class_correct=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
class_total = list(0. for i in range(10))  # class_total=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)  # outputs的维度是：4*10
        # torch.max(outputs.data, 1)返回outputs每一行中最大值的那个元素，且返回其索引
        # 此时predicted的维度是：4*1
        _, predicted = torch.max(outputs, 1)
        # 此时c的维度：4将预测值与实际标签进行比较，且进行降维
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
#保存权重和参数
save_path = 'Lenets.pth'#与当前.py文件同级
torch.save(net.state_dict(), save_path)