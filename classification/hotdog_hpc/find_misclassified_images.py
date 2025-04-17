import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from architectures.VGG16 import VGG16
import wandb
import copy
from experiment import Experiment

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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VGG16().to(device)
initial_model = copy.deepcopy(model)

num_epochs = 30
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()

# 创建保存图像的文件夹
save_dir = 'optimizer_performance_plots'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 保存时间和最佳验证准确率的 Excel 文件路径
excel_file = 'training_times.xlsx'
# 列表用于存储每个模型的结果
training_data = []

# 用于存储每个epoch的loss和accuracy
train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 20)

    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    # 每个 epoch 有训练和验证阶段
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # 训练模式
            data_loader = train_loader
        else:
            model.eval()  # 验证模式
            data_loader = test_loader

        running_loss = 0.0
        running_corrects = 0

        # 遍历数据
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            # labels = labels.view(-1,1).float()  # 确保标签为float型
            labels = labels.to(device)

            # 在训练模式下，清空梯度
            if phase == 'train':
                optimizer.zero_grad()

            # 前向传播
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                preds = (torch.sigmoid(outputs) > 0.5).float()  # 二元分类需要用 sigmoid
                loss = criterion(outputs, labels.unsqueeze(1).float())
                if phase == 'val' and epoch == num_epochs - 1:
                    misclassified = preds.squeeze() != labels
                    if misclassified.any():
                        misclassified_images.append(inputs[misclassified])
                        misclassified_labels.append(labels[misclassified])
                        misclassified_preds.append(preds[misclassified])

                # 在训练模式下，反向传播和优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.unsqueeze(1))

        epoch_loss = running_loss / len(data_loader.dataset)
        epoch_acc = running_corrects.double() / len(data_loader.dataset)

        if phase == 'train':
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
        else:
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc)

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

# 将错误分类的图片、标签、预测值转为列表
misclassified_images = torch.cat(misclassified_images, 0).cpu()
misclassified_labels = torch.cat(misclassified_labels, 0).cpu()
misclassified_preds = torch.cat(misclassified_preds, 0).cpu()

save_dir = 'misclassified_images'
# 如果要保存到文件夹中
if save_dir and not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 遍历并显示或保存错误分类的图片
for i in range(len(misclassified_images)):
    image = misclassified_images[i].permute(1, 2, 0).numpy()  # 将图像转换为 numpy 格式
    true_label = misclassified_labels[i].item()
    pred_label = misclassified_preds[i].item()
    plt.imshow(image)
    plt.title(f'True: {true_label}, Pred: {pred_label}')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, f'misclassified_{i}.png'))

    