import time
import torch
from architectures.ResNet18 import ResNet18
from architectures.VGG16 import VGG16
from architectures.diy1 import DIY1
from architectures.diy2 import DIY2
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy

class Experiment():
    def __init__(
            self,
            architectures,
            train_loader,
            test_loader,
            lr,
            optimizer,
            augmentations,
            num_epochs,
            ):
        self.architectures = architectures
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lr = lr
        self.optimizer = optimizer
        self.augmentations = augmentations
        self.num_epochs = num_epochs
    
    
    def _init_model(self,device):
        if self.architectures == 'vgg16':
            self.model = VGG16().to(device)
        elif self.architectures == 'resnet18':
            self.model = ResNet18(num_classes=1).to(device)
        elif self.architectures == 'diy1':
            self.model = DIY1().to(device)
        elif self.architectures == 'diy2':
            self.model = DIY2().to(device)

    def _init_optimizer(self):
        combined_optimizer = {}  # {name: optimizer}
        for lr in self.lr:
            for opt in self.optimizer:
                if opt == 'adam':
                    combined_optimizer[f'adam with lr={lr}'] = torch.optim.Adam(self.model.parameters(), lr=lr)
                elif opt == 'sgd':
                    combined_optimizer[f'SGD with lr={lr}'] = torch.optim.SGD(self.model.parameters(), lr=lr)
                elif opt == 'sgd+momentum':
                    combined_optimizer[f'SGD+momentum with lr={lr}'] = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
                else:
                    raise ValueError('Optimizer not supported')
        return combined_optimizer

    # write the function to train the model
    def train_model(self):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._init_model(device)
        self.initial_model = copy.deepcopy(self.model)
        self.combined_optimizer = self._init_optimizer()
        # 二元分类的损失函数
        criterion = torch.nn.BCEWithLogitsLoss()

        # 创建保存图像的文件夹
        save_dir = 'optimizer_performance_plots'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # 保存时间和最佳验证准确率的 Excel 文件路径
        excel_file = 'training_times.xlsx'
        # 列表用于存储每个模型的结果
        training_data = []

        # 遍历 combined_optimizer 字典中的每个优化器
        for opt_name, optimizer in self.combined_optimizer.items():
            print(f'Training with optimizer: {opt_name}')
            self.model.load_state_dict(self.initial_model.state_dict())

            since = time.time()

            best_model_wts = self.model.state_dict()
            best_acc = 0.0

            # 用于存储每个epoch的loss和accuracy
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []

            for epoch in range(self.num_epochs):
                print(f'Epoch {epoch + 1}/{self.num_epochs}')
                print('-' * 20)

                # 每个 epoch 有训练和验证阶段
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.model.train()  # 训练模式
                        data_loader = self.train_loader
                    else:
                        self.model.eval()  # 验证模式
                        data_loader = self.test_loader

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
                            outputs = self.model(inputs)
                            preds = (torch.sigmoid(outputs) > 0.5).float()  # 二元分类需要用 sigmoid
                            loss = criterion(outputs, labels.unsqueeze(1).float())

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

                    # 在验证集上检查最佳模型
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = self.model.state_dict()


            # 训练结束后加载最佳模型权重
            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:.4f}')

            self.model.load_state_dict(best_model_wts)

            # 绘制损失和准确率曲线
            save_path = os.path.join(save_dir, f'{opt_name}_performance.png')
            plot_performance(train_losses, val_losses, train_accs, val_accs, opt_name, save_path)

            # 保存训练时间和最佳验证准确率
            training_data.append({
                'Model': opt_name,
                'Training Time (s)': time_elapsed,
                'Best Validation Accuracy': best_acc.item()
            })

            # 绘制损失和准确率曲线
            save_path = os.path.join(save_dir, f'{opt_name}_performance.png')
            plot_performance(train_losses, val_losses, train_accs, val_accs, opt_name, save_path)
            
        print("Training Data:", training_data)
        # 将数据保存到 Excel 文件
        save_to_excel(training_data, excel_file)
        return self.model

        # 主函数：在 3x3 网格中展示性能分析图
    
    def plot_all_optimizers_performance(self):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.ravel()  # 将3x3的axes平展为1维

        for idx, (opt_name, optimizer) in enumerate(self.combined_optimizer.items()):
            train_losses = []
            val_losses = []
            train_accs = []
            val_accs = []

            # 遍历每个 epoch，收集损失和准确率（你可以调用 train_model 的部分代码）
            # 假设这里你已经完成了训练过程，保存了每个 epoch 的loss和accuracy

            # 在网格中绘制每个优化器的损失和准确率图
            epochs = range(1, len(train_losses) + 1)
            axes[idx].plot(epochs, train_losses, label='Train Loss')
            axes[idx].plot(epochs, val_losses, label='Validation Loss')
            axes[idx].plot(epochs, train_accs, label='Train Accuracy')
            axes[idx].plot(epochs, val_accs, label='Validation Accuracy')
            axes[idx].set_title(f'{opt_name}')
            axes[idx].legend()

        plt.tight_layout()
        plt.show()

# 绘制损失和准确率图的函数
def plot_performance(train_losses, val_losses, train_accs, val_accs, opt_name,save_path):
    epochs = range(1, len(train_losses) + 1)

    # 检查输入数据是否正确
    print(f"Train Losses: {train_losses}")
    print(f"Validation Losses: {val_losses}")
    print(f"Train Accuracy: {train_accs}")
    print(f"Validation Accuracy: {val_accs}")

    # 将 GPU Tensor 转换为 CPU NumPy 数组
    train_accs = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in train_accs]
    val_accs = [acc.cpu().numpy() if isinstance(acc, torch.Tensor) else acc for acc in val_accs]
    train_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    val_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in val_losses]


    plt.figure(figsize=(12, 6))

    # 绘制loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title(f'{opt_name} Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 绘制accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.title(f'{opt_name} Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    # 保存图像到指定路径
    plt.savefig(save_path)
    print(f'Saved plot to {save_path}')
    plt.close()  # 关闭当前图，以释放内存

def save_to_excel(data, excel_file):
    df = pd.DataFrame(data)

    if os.path.exists(excel_file):
        # 如果文件存在，则追加数据
        existing_df = pd.read_excel(excel_file)
        df = pd.concat([existing_df, df], ignore_index=True)

    # 保存到 Excel 文件
    df.to_excel(excel_file, index=False)
    print(f'Saved training times and validation accuracy to {excel_file}')

def reset_model_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()