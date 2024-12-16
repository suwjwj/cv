'''
注意，上传文件服务器时，将model.py、best_model.pt、infer.py三个文件，
放置于名为‘’z学'的文件夹下，上传文件夹
'''
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.metrics import accuracy_score
from models import *


# 教师端的测试集路径
path_img = '../datasets/cats-images/test'
# 预训练模型文件路径，放置于'z学号'目录下。
# 注意：是整个模型的文件，不是模型权重参数文件
path_model = './z04222809/best_cats_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 下面的语句替换为自己的transforms设置
transform = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

#导入模型
model = torch.load(path_model)
# 导入测试集
data = ImageFolder(path_img, transform=transform)
classes = data.classes
print(classes)
# 上传时batch_size设置为60, 教师的测试集为60张图片
test_data = DataLoader(data, batch_size=60, shuffle=False)

# 开始测试
model.to(device)
model.eval()

with torch.no_grad():
    x, y = next(iter(test_data))
    x = x.to(device)
    out = model(x)
    
    # 使用Sigmoid输出的阈值判断
    predicts = torch.gt(out, 0.5).squeeze()
    
    acc = accuracy_score(y.cpu(), predicts.cpu())
    print(f'Test Accuracy: {acc*100.0:.2f}%')



