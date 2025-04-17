import time
import torch
from architectures.ResNet18 import ResNet18
from architectures.VGG16 import VGG16
import matplotlib.pyplot as plt
import os
import pandas as pd
import copy
import torch.nn as nn
import numpy as np
from PIL import Image


from matplotlib import pyplot as plt
import torch.nn.functional as F
import matplotlib.cm as cm
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

# 确保 shortcuts_saliency_maps 文件夹存在
save_dir = 'shortcuts_saliency_maps'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def normalize(saliency: torch.Tensor):
    return (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

def generate_smoothed_image(image, num_samples, sigma):
  """Generate smoothed images with noise

  Args:
      image (torch.Tensor): Image to be smoothed
      num_samples (int): Number of smoothed images to generate
      sigma (float): Standard deviation of the noise

  Returns:
      torch.Tensor: Smoothed images
  """
  images = []
  for _ in range(num_samples):
      noise = torch.randn_like(image) * sigma
      noisy_image = image + noise
      noisy_image = torch.clamp(noisy_image, 0, 1)
      images.append(noisy_image)
  return torch.stack(images, dim=0)

def compute_smooth_grad(model, image, target, num_samples=50, sigma=0.1):
    """Compute the smooth gradient of the image

    Args:
        model (torch.nn.Module): Model to be used
        image (torch.Tensor): Image to be smoothed
        target (torch.Tensor): Target of the image
        num_samples (int): Number of smoothed images to generate
        sigma (float): Standard deviation of the noise

    Returns:
        torch.Tensor: Smoothed images
    """
    imgs = generate_smoothed_image(image, num_samples, sigma)
    model.eval()
    # track differentiation
    imgs.requires_grad = True
    outputs = model(imgs)
    loss = outputs[:, target].sum()
    model.zero_grad()
    loss.backward()

    smooth_grad = imgs.grad.data.mean(dim=0, keepdim=True)
    # No matter the sign, we take the absolute value
    # since we only care about the magnitude of the influence
    # of each pixel on the prediction, not its direction
    saliency = smooth_grad.abs().squeeze().detach().cpu().numpy()

    if saliency.shape[0] == 3:
        saliency = np.sum(saliency, axis=0)

    noise_level = sigma / (imgs.max() - imgs.min())

    return normalize(saliency), noise_level

def generate_baseline_images(image, baseline, num_samples=50):
    """Generate baseline images

    Args:
        image (torch.Tensor): Image to be smoothed
        num_samples (int): Number of baseline images to generate
    """
    images = []
    factor = np.linspace(0, 1, num_samples+1)
    for a in factor:
      img = baseline * a * (image - baseline)
      images.append(img)
    return torch.stack(images, dim=0)

def compute_integrated_gradients(model, image, target, baseline = None, num_samples=50):
    """Compute the integrated gradients of the image

    Args:
        model (torch.nn.Module): Model to be used
        image (torch.Tensor): Image to be smoothed
        target (torch.Tensor): Target of the image
        num_samples (int): Number of smoothed images to generate
    """
    if baseline is None:
        baseline = torch.zeros_like(image)
    imgs = generate_baseline_images(image, baseline, num_samples)
    print(imgs.shape) # torch.Size([51, 3, 64, 64])
    print(target.shape) # torch.Size([])
    imgs.requires_grad = True
    model.eval()

    outputs = model(imgs)
    loss = outputs[0, target]
    print('outputs',outputs.shape) # outputs torch.Size([51, 2])
    print('loss',loss.shape) # loss torch.Size([])
    model.zero_grad()
    loss.backward()

    integrated_grads = imgs.grad.data.mean(dim=0, keepdim=True)
    integrated_grads = (image - baseline) * integrated_grads
    integrated_grads = integrated_grads.abs().squeeze().detach().cpu().numpy()

    if integrated_grads.ndim == 3:
        integrated_grads = np.mean(integrated_grads, axis=0)

    # normalize saliency map
    return normalize(integrated_grads)

def plot_saliency_map(img, smooth_grad, integrated_gradients,path = None):
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img.squeeze().permute(1, 2, 0))
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(smooth_grad, cmap=plt.cm.gray)
    plt.title('Smooth Grad')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(integrated_gradients, cmap=plt.cm.gray)
    plt.title('Integrated Gradients')
    plt.axis('off')
    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

def get_conv_layer(model, n = 1) -> tuple[nn.Module, int]:
    """Get the last n-th convolutional layer of the model

    Args:
        model (torch.nn.Module): Model to be used
        n (int): The n-th convolutional layer to get

    Returns:
        torch.nn.Module: Last convolutional layer
        int: Index of the last convolutional layer
    """
    # iterate through all layers
    count = 1
    for i in range(len(model.net) - 1, -1, -1):
      if isinstance(model.net[i], nn.Conv2d):
        if count == n:
          return model.net[i], i
        count += 1
    return None, None

def compute_grad_cam(model, image, target):
    """Compute the grad cam of the image

    Args:
        model (torch.nn.Module): Model to be used
        image (torch.Tensor): Image to be smoothed
        target (torch.Tensor): Target of the image
    """
    image = image.unsqueeze(0)
    last_conv_layer, _ = get_conv_layer(model, 1) # sometime the last 2 or 3 layer better

    gradients = {}
    activations = {}

    image.requires_grad = True
    model.eval()
    def forward_hook(module, input, output):
        activations['value'] = output
    last_conv_layer.register_forward_hook(forward_hook)

    def backward_hook(module, input, output):
        gradients['value'] = output[0]
    last_conv_layer.register_full_backward_hook(backward_hook)

    output = model(image)
    loss = output[:, target].sum()
    model.zero_grad()
    loss.backward()

    grads = gradients['value']
    fmap = activations['value']

    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    grad_cam = torch.sum(weights * fmap, dim=1)
    grad_cam = torch.relu(grad_cam)
    grad_cam = grad_cam.squeeze(0).detach().cpu().numpy()

    return normalize(grad_cam)

def plot_saliency_hot_map(img, grad_cam, path = None):
    if img.dim() == 4:
        img = img.squeeze(0)
    image_np = img.detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))  # [H, W, C]
    if image_np.max() > 1:
        image_np = image_np / 255.0

    grad_cam_tensor = torch.from_numpy(grad_cam).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    height, width = image_np.shape[:2]
    grad_cam_resized = F.interpolate(grad_cam_tensor, size=(height, width), mode='bilinear', align_corners=False)
    grad_cam_resized = grad_cam_resized.squeeze().detach().cpu().numpy()

    colormap = plt.get_cmap('jet')
    heatmap = colormap(grad_cam_resized)[:, :, :3]

    alpha = 0.5
    overlay = heatmap * alpha + image_np * (1 - alpha)
    overlay = np.clip(overlay, 0, 1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title('Grad Cam')
    plt.axis('off')

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
        plt.close()

# 绘制并保存 Saliency Maps
def plot_saliency_maps(img, smooth_grad, integrated_gradients, grad_cam, img_name):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(img.squeeze().permute(1, 2, 0))
    axs[0].set_title('Original Image')
    axs[1].imshow(smooth_grad, cmap='gray')
    axs[1].set_title('Smooth Grad')
    axs[2].imshow(integrated_gradients, cmap='gray')
    axs[2].set_title('Integrated Gradients')
    axs[3].imshow(grad_cam, cmap='jet', alpha=0.5)
    axs[3].set_title('Grad-CAM')
    for ax in axs:
        ax.axis('off')

    save_path = os.path.join(save_dir, f'{img_name}_saliency_maps.png')
    plt.savefig(save_path)
    plt.close()
    print(f'Saved saliency maps for {img_name}')


class SC_Experiment():
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

        # 获取图像路径
        image_folder = os.path.join('hotdog_nothotdog', 'train', 'hotdog')
        image_paths = sorted(os.listdir(image_folder))[:10]  # 选取前 10 个图像
        image_paths = [img for img in image_paths if img.endswith(('.jpg', '.jpeg', '.png'))]

        for img_name in image_paths:
            img_path = os.path.join(image_folder, img_name)
            image = Image.open(img_path)
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor()
            ])
            transform2 = transforms.Compose([
                transforms.ToTensor()
            ])
            img = transform(image)
            # 目标类别 (例如，假设目标类别为 0)
            target = torch.tensor(0)

            # 计算三种解释方法
            grad_cam = compute_grad_cam(self.model, img.clone().detach(), target)
            saliency = compute_smooth_grad(self.model, img.clone().detach(), target)
            integrated_gradients = compute_integrated_gradients(self.model, img.clone().detach(), target)

            # 保存结果
            plot_saliency_maps(img, saliency, integrated_gradients, grad_cam, img_name)

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