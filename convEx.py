import sys
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
image_path = "img.png"
image = Image.open(image_path).convert("RGB")
transform = torchvision.transforms.ToTensor()
image_tensor = transform(image).unsqueeze(0)
def box(n):
    return np.ones((n, n), dtype=np.float32) / (n * n)
def conv(type):
    if type == "sobelY":
        kernel = [
            [1,2,1],
            [0,0,0],
            [-1,-2,-1]
        ]
    elif type == "box3":
        kernel = [
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
        [1/9, 1/9, 1/9]
        ]
    elif type == "box5":
        kernel = box(5)
    elif type == "box7":
        kernel = box(7)
    elif type == "box9":
        kernel = box(9)
    elif type == "sobelX":
        kernel = [
            [1,0,-1],
            [2, 0, -2],
            [1, 0, -1]
        ]

    kernel_size = 50
    kernel_value = 1.0 / (kernel_size * kernel_size)
    #kernel = torch.ones((3, 1, kernel_size, kernel_size)) * kernel_value  # Shape: [3, 1, 3, 3]

    kernelTens = torch.tensor([[kernel],[kernel],[kernel]],dtype=torch.float32)
    #kernel = np.ones((3, 3)) / 9
    blurredImage = torch.nn.functional.conv2d(image_tensor, kernelTens,stride=1,padding=0,groups=3)
    blurredImage= torchvision.transforms.ToPILImage()(blurredImage.squeeze(0))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.title('Kernel')
    plt.imshow(kernel)

    plt.subplot(1, 3, 3)

    plt.title('Result')
    plt.imshow(blurredImage, )
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    conv("sobelX")
    conv("sobelY")

    globals()[sys.argv[1]](sys.argv[2])