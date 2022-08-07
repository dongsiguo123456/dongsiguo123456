import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
# （ToTensor()：把一个PIL.Image转换成Tensor，Normalize()：标准化，即减均值，除以标准差）
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([480,640])])
#label是按照文件夹名顺序排序后存成字典，即{类名:类序号(从0开始)}
#100个类即100个文件夹即100个人的各有4个照片，所以lable分为100个类即文件夹的命名，idx从序号0开始到99结束
dataset=ImageFolder('./dataset',transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, test_size])

print(dataset.class_to_idx)

