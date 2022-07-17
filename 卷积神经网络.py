import torch
from torch import nn
from torch.nn import functional as F
#nn.中的类需要先初始化，在写forward函数对类中的函数进行调用

class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5,self).__init__()
        self.conv_unit=nn.Sequential(
            #b代表第几张照片
            # [b,3,32,32]=>[b,6,
            nn.Conv2d(3,6,kernel_size=5,stride=1,padding=0),
            #AvgPOOL2D 不改变out_channels的大小
            nn.AvgPool2d(kernel_size=2,stride=2,padding=0),
            # 第二个卷积层，16为输出out_channels的大小，数目变多
            nn.Conv2d(6,16,kernel_size=5,stride=1,padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),


        )
        self.fc_unit=nn.Sequential(
            #打平，变为一个维度
            nn.Linear(16*5*5,120),
            #Relu是激活函数
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        )
        temp=torch.randn(2,3,32,32)
        out=self.conv_unit(temp)
        #打印的值为[2,16,5,5]
        print("conv",out.shape)
        #分类问题一般使用交叉熵
        self.criteon=nn.CrossEntropyLoss()
    def forward(self,x):
        # [b,3,32,32]=>[b,16,5,5]
        batchsz=x.size(0)
        x=self.conv_unit(x)
        #x.view()改变维度
        x=x.view(batchsz,16*5*5)
        #[b,16*5*5]=>[b,10]
        logits=self.fc_unit(x)
        # [b.10]
        # pred=F.softmax(logits,dim=1)
        # loss=self.criteon(logits，y)
        return logits
       
def main():
    net=Lenet5()
    temp=torch.randn(2,3,32,32)
    out=net(temp)
    #打印的值为[2,16,5,5]
    print("conv",out.shape)


if __name__=='__main__':
    main()

