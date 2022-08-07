import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class facenet(nn.Module):
    def __init__(self):
        super().__init__()

        #首先需要先定义几个模型
        self.conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=(3,3),padding=1)
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),padding=1)
        self.conv3=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv4=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn1=nn.BatchNorm2d(3)
        self.bn2=nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(64)
        #nn.MaxPool2d(2,2)可以将传入的大小变为原来的一半
        self.maxpool=nn.MaxPool2d((2,2),2)
        self.fc1=nn.Linear(in_features=64*30*40,out_features=1024)
        #最后100是有100个类，再用softmax函数即可得出100个类的概率
        self.fc2 = nn.Linear(in_features=1024, out_features=100)

    def forward(self,x):
        x=self.bn1(x)
        x=F.relu(self.conv1(x))
        x=self.maxpool(x)
        x=F.relu(self.conv2(x))
        x=self.maxpool(x)
        x=self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x=self.bn3(x)
        x=x.view(-1,64*30*40)
        x=F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        print(x.shape)#最后形状大小为torch.Size([1, 100])
        return x




if __name__=='__main__':
    test_data=torch.randn(1,3,480,640)
    net=facenet()
    output=net(test_data)