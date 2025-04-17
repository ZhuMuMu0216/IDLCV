from matplotlib import pyplot as plt
import torch.nn.functional as F
import matplotlib.cm as cm
import torch
import numpy as np
from data_loader import HotdogDataLoader
import torch.multiprocessing as mp
from models import VGG16
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import os

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

def main():
    model = VGG16(num_classes=2)
    model.load_state_dict(torch.load('./assignment1/results/model.pth', map_location=torch.device('cpu')))
    dl = HotdogDataLoader(64, False, 64, 0.2)
    train_loader, _, _, _, _, _ = dl.data_for_exp()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'hotdog_nothotdog', 'train', 'hotdog', 'chilidog (1).jpg')
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor()
    ])
    img = transform(image)
    target = torch.tensor(0)

    grad_cam = compute_grad_cam(model, img.clone().detach(), target)
    saliency, noise_level = compute_smooth_grad(model, img.clone().detach(), target)
    integrated_gradients = compute_integrated_gradients(model, img.clone().detach(), target)
    print('noise_level', noise_level)
    plot_saliency_map(img, saliency, integrated_gradients)
    plot_saliency_hot_map(transform2(image), grad_cam)




if __name__ == '__main__':
    mp.freeze_support()  # This is necessary for Windows compatibility
    main()
