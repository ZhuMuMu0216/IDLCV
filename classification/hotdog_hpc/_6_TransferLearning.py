import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class Hotdog_NotHotdog(torch.utils.data.Dataset):
    def __init__(self, train, transform, data_path='hotdog_nothotdog'):
        'Initialization'
        self.transform = transform
        data_path = os.path.join(data_path, 'train' if train else 'test')
        image_classes = [os.path.split(d)[1] for d in glob.glob(data_path +'/*') if os.path.isdir(d)]
        image_classes.sort()
        self.name_to_label = {c: id for id, c in enumerate(image_classes)}
        self.image_paths = glob.glob(data_path + '/*/*.jpg')
        
    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)

    def __getitem__(self, idx):
        'Generates one sample of data'
        image_path = self.image_paths[idx]
        
        image = Image.open(image_path)
        c = os.path.split(os.path.split(image_path)[0])[1]
        y = self.name_to_label[c]
        X = self.transform(image)
        return X, y

size = 128
train_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])
test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

# batch_size = 32
# trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
# train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
# testset = Hotdog_NotHotdog(train=False, transform=test_transform)
# test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# HPC version
batch_size = 64
trainset = Hotdog_NotHotdog(train=True, transform=train_transform)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testset = Hotdog_NotHotdog(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)


'''
Below we start train two models: one with pretrained weights and one without.
And compare their performance.
'''


# 定义两个模型，一个带预训练参数，一个不带预训练参数
model_without_pretrained = models.resnet18(pretrained=False)
model_with_pretrained = models.resnet18(pretrained=True)

# 冻结 ResNet18 的所有参数，以防止它们在训练过程中更新
for param in model_with_pretrained.parameters():
    param.requires_grad = False

# 替换最后的全连接层，使其适应二分类任务
num_ftrs = model_with_pretrained.fc.in_features
model_with_pretrained.fc = nn.Linear(num_ftrs, 1)  # 二分类
model_without_pretrained.fc = nn.Linear(num_ftrs, 1)  # 二分类

# 将模型移动到 GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_with_pretrained = model_with_pretrained.to(device)
model_without_pretrained = model_without_pretrained.to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类可以使用 BCEWithLogitsLoss

# 定义不同的优化器
optimizer_pretrained = torch.optim.Adam(model_with_pretrained.fc.parameters(), lr=0.01)
optimizer_without_pretrained = torch.optim.Adam(model_without_pretrained.parameters(), lr=0.01)

# 初始化数据加载器（假设已经定义好了 train_loader 和 test_loader）
num_epochs = 30

# 用于记录损失和准确率的列表
train_loss_pretrained = []
train_loss_without_pretrained = []
val_loss_pretrained = []
val_loss_without_pretrained = []

train_acc_pretrained = []
train_acc_without_pretrained = []
val_acc_pretrained = []
val_acc_without_pretrained = []


# 开始训练和验证
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 20)

    # 两个模型的训练和验证阶段
    for model, optimizer, train_loss, val_loss, train_acc, val_acc, model_name in [
        (model_with_pretrained, optimizer_pretrained, train_loss_pretrained, val_loss_pretrained, train_acc_pretrained, val_acc_pretrained, "Pretrained Model"),
        (model_without_pretrained, optimizer_without_pretrained, train_loss_without_pretrained, val_loss_without_pretrained, train_acc_without_pretrained, val_acc_without_pretrained, "Non-Pretrained Model")
    ]:
        best_acc = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = test_loader

            running_loss = 0.0
            running_corrects = 0

            # 遍历数据
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # 清空梯度
                if phase == 'train':
                    optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = (torch.sigmoid(outputs) > 0.5).float()  # 二元分类需要用 sigmoid
                    loss = criterion(outputs, labels.unsqueeze(1).float())

                    # 在训练阶段进行反向传播和优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计损失和准确率
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.unsqueeze(1))

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            # 将每个阶段的损失和准确率保存
            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc.item())
            else:
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc.item())

            print(f'{model_name} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# 定义保存图像的文件夹路径
save_dir = 'TransferLearningCompare'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # 如果文件夹不存在，创建文件夹

# 绘制损失曲线
plt.figure(figsize=(10, 5))

# 训练和验证损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_loss_pretrained, label='Pretrained Train Loss')
plt.plot(val_loss_pretrained, label='Pretrained Val Loss')
plt.plot(train_loss_without_pretrained, label='Non-Pretrained Train Loss')
plt.plot(val_loss_without_pretrained, label='Non-Pretrained Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 训练和验证准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_acc_pretrained, label='Pretrained Train Acc')
plt.plot(val_acc_pretrained, label='Pretrained Val Acc')
plt.plot(train_acc_without_pretrained, label='Non-Pretrained Train Acc')
plt.plot(val_acc_without_pretrained, label='Non-Pretrained Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()

# 保存图片到 TransferLearningCompare 文件夹中
save_path = os.path.join(save_dir, 'TransferLearning_Comparison.png')
plt.savefig(save_path)
print(f'Saved plot to {save_path}')

# 显示图像
plt.show()