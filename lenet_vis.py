#!/usr/bin/env python3


import torch
from torchvision.datasets import mnist
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


if __name__ == '__main__':
    layer_outputs = {}

    #hooks
    def hook(module, input, output):
        layer_outputs[module] = output
    # load trained model
    lenet5 = torch.load("./trained_models/MNIST_epoch43_acc0.9913.pkl")
    lenet5.eval()
    for name, layer in lenet5.named_modules():
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ReLU) or isinstance(layer,
                                                                                                torch.nn.MaxPool2d) or isinstance(
                layer, torch.nn.AdaptiveAvgPool2d):
            layer.register_forward_hook(hook)
    # load test dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((32,32)),
                                    transforms.Normalize(mean=(0.5,), std=(0.5,))
                                   ])


    method_to_call = mnist.MNIST
    dataset = method_to_call(root="./data", train=False, transform=transform, download=True)

    test_loader = DataLoader(dataset, 500,)
    
    with torch.no_grad():
        for idx, (test_x, test_label) in enumerate(test_loader):
            plt.imshow(test_x[0][0],cmap='gray')
            plt.show()
            predict_y = lenet5(test_x.float())
            break

    def visualize_feature_maps(layer_outputs):
        for i, (layer, output) in enumerate(layer_outputs.items()):
            if i >= 120:
                continue

            fig, axes = plt.subplots(1,output.shape[1] , figsize=(20, 6))
            for idx in range(output.shape[1]):
                if idx < output.shape[1]:
                    feature_map = output[0, idx].cpu().numpy()
                    axes[idx].imshow(feature_map)
                axes[int(idx)].axis('off')
            fig.suptitle(f"Layer: {layer}")
            plt.tight_layout()
            plt.show()



    visualize_feature_maps(layer_outputs)
    kernels = lenet5.conv1.weight.data.clone()

    fig, axes = plt.subplots(1, 6, figsize=(12, 12))
    vmin = kernels.min().item()
    vmax = kernels.max().item()
    for i, ax in enumerate(axes.flat):
        if i < kernels.shape[0]:
            kernel = kernels[i]
            img = ax.imshow(kernel.permute(1, 2, 0), vmin=vmin, vmax=vmax)
        ax.axis('off')
    plt.show()
