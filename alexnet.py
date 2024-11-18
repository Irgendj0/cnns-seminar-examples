import math

import torch

import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

layer_outputs = {}


def hook(module, input, output):
    layer_outputs[module] = output
#hooks
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.MaxPool2d) or isinstance(layer, torch.nn.AdaptiveAvgPool2d):
        layer.register_forward_hook(hook)

def preprocess_image(image_path):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch


image_path = "samoyed.jpg"
input_batch = preprocess_image(image_path)
with torch.no_grad():
    res = model(input_batch)
def visualize_feature_maps(layer_outputs):
    for i, (layer, output) in enumerate(layer_outputs.items()):
        #only first layer, can be changed
        if i >= 1:
            continue

        cols = 4
        fig, axes = plt.subplots(cols, int(output.shape[1]/cols)+1, figsize=(20, 6))
        for idx in range(cols*(int(output.shape[1]/cols)+1)):
            if idx < output.shape[1]:
                feature_map = output[0, idx].cpu().numpy()
                axes[int(idx%cols),int(idx/cols)].imshow(feature_map)
            axes[int(idx%cols),int(idx/cols)].axis('off')
        fig.suptitle(f"Layer: {layer}")
        plt.tight_layout()
        plt.show()

visualize_feature_maps(layer_outputs)


kernels = model.features[0].weight.data.clone()
kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

fig, axes = plt.subplots(8, 8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i < kernels.shape[0]:

        kernel = kernels[i]
        ax.imshow(kernel.permute(1, 2, 0))
    ax.axis('off')

plt.show()