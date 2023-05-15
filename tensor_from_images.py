import os

import imageio.v2 as imageio
import torch

if __name__ == '__main__':
    image_dir = './image_data'
    image_names = [image_name for image_name in os.listdir(image_dir)
                   if os.path.splitext(image_name)[-1] == '.bmp']
    batch_size = len(image_names)
    batch = torch.zeros(batch_size, 3, 256, 256)
    for i, image_name in enumerate(image_names):
        image_array = imageio.imread(os.path.join(image_dir, image_name))
        image_tensor = torch.tensor(image_array)
        image_tensor = image_tensor.permute(2, 0, 1)
        image_tensor = image_tensor[:3]
        batch[i] = image_tensor
    brightness = batch.mean(dim=(1, 2, 3))
    rgb_brightness = batch.mean(dim=(2, 3))
    for i, image_name in enumerate(image_names):
        print(f'{i}: {image_name}, {rgb_brightness[i]}')
