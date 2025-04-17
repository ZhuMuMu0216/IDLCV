import os
from PIL import Image, ImageDraw
import random

data_path1 = ['hotdog_nothotdog/test/hotdog',
             'hotdog_nothotdog/train/hotdog',]
             


for i in data_path1 :
    # 定义图片路径和保存路径
    data_path = i
    save_path = data_path
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 获取hotdog文件夹中所有图片的路径
    image_paths = sorted(os.listdir(data_path))


    # 计算要添加高亮的图片数量（66%）
    num_images_to_highlight = int(len(image_paths) * 0.66)


    # 随机选择66%的图片
    highlighted_image_paths = random.sample(image_paths, num_images_to_highlight)

    # 定义一个函数，在右上角加上一个高亮区域
    def add_highlight(image, size=25, color=(255, 0, 0)):
        """在图像右上角添加一个高亮区域"""
        draw = ImageDraw.Draw(image)
        # 计算高亮方块的区域（右上角）
        width, height = image.size
        draw.rectangle([width - size, 0, width, size], fill=color)
        return image

    # 遍历所有图片，对选中的50%图片加上高亮
    for i, img_name in enumerate(image_paths):
        img_path = os.path.join(data_path, img_name)
        
        try:
            # 打开图片
            image = Image.open(img_path)

            # 如果图片在被选中的50%列表中，加上高亮
            if img_name in highlighted_image_paths:
                image_with_highlight = add_highlight(image)
                print(f"Processed and saved {img_name} with highlight.")
            else:
                image_with_highlight = image  # 不修改的图片保持原样

            # 保存新的图片
            image_with_highlight.save(os.path.join(save_path, img_name))
            
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print("Finished processing images.")