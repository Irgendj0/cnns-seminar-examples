# cnns-seminar-examples
Contains examples used in a seminar on CNNs.
These files are used to illustrate how CNNs work. Outputs are in general done using matplotlib. I suggest you use an IDE for that project.
# Usage
## conv1d.py
```
python conv1d.py
```
Generates a gif showing the convolution of two rectangles
## convEx.py
```
python convEx.py conv box3 #<- select kernel
```
Applies a 2d convolution to an image. Available are: box3, box5,box7,box9,sobelX,sobelY
## lenet_vis.py
```
python lenet_vis.py
```
Shows the outputs of the intermediate layers and kernels of the first layer of LeNet-5. Based on [this](https://github.com/hrfang/LeNet5-code-examples) repo and pretrained model.
## alexnet.py
```
python alexnet.py
```
Shows the outputs of the intermediate layers and kernels of the first layer of AlexNet. Uses the pretrained model available in pytorch.
# Note
This code is distributed without warranty. Use at your own risk.
