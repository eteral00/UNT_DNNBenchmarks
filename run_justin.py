''' run.py
Executes an inference for images stored within the local "data" directory, 
yielding a collection of named .bin files representing inputs/outputs to 
convolution layers. 

Usage: 
  python3 run.py <model type> 

Model types: 
  -resnet18 
  -vgg16 
  -alexnet
  -yolov5l
'''

import sys 
import math
import numpy as np
import os
import subprocess
import torch

from imagenet_stubs.imagenet_2012_labels import label_to_name
from glob import glob
from torch import nn
from torch.nn import functional as F
from torchvision import models as M

from PIL import Image


class resnet18:
    def __init__(self): 
        self.model = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1).eval()
        
    
    # Returns (x, control) 
    def __call__(self, dataset):
        resnet18 = self.model 
        x = dataset 
        control = resnet18(x.clone()) 
    
        ## Khoa
        resultFilePath = "cL0_1_1_"
        x = conv2d(x, resultFilePath, resnet18.conv1)
        x = resnet18.bn1(x)
        x = nn.ReLU(inplace=True)(x)
        x = resnet18.maxpool(x)

        # LAYER 1

        identity = x
        resultFilePath = "cL1_0_1_"
        x = conv2d(identity, resultFilePath, resnet18.layer1[0].conv1)
        x = resnet18.get_submodule("layer1.0.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL1_0_2_"
        x = conv2d(x, resultFilePath, resnet18.layer1[0].conv2)
        x = resnet18.get_submodule("layer1.0.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        identity = x

        resultFilePath = "cL1_1_1_"
        x = conv2d(x, resultFilePath, resnet18.layer1[1].conv1)
        x = resnet18.get_submodule("layer1.1.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL1_1_2_"
        x = conv2d(x, resultFilePath, resnet18.layer1[1].conv2)
        x = resnet18.get_submodule("layer1.1.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        # LAYER 2

        identity = x

        resultFilePath = "cL2_0_1_"
        x = conv2d(x, resultFilePath, resnet18.layer2[0].conv1)
        x = resnet18.get_submodule("layer2.0.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL2_0_2_"
        x = conv2d(x, resultFilePath, resnet18.layer2[0].conv2)
        x = resnet18.get_submodule("layer2.0.bn2")(x)

        resultFilePath = "cL2_0_down_"
        identity = conv2d(identity, resultFilePath, resnet18.get_submodule("layer2.0.downsample.0"))
        identity = resnet18.get_submodule("layer2.0.downsample.1")(identity)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        identity = x

        resultFilePath = "cL2_1_1_"
        x = conv2d(x, resultFilePath, resnet18.layer2[1].conv1)
        x = resnet18.get_submodule("layer2.1.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL2_1_2_"
        x = conv2d(x, resultFilePath, resnet18.layer2[1].conv2)
        x = resnet18.get_submodule("layer2.1.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        # LAYER 3

        identity = x

        resultFilePath = "cL3_0_1_"
        x = conv2d(x, resultFilePath, resnet18.layer3[0].conv1)
        x = resnet18.get_submodule("layer3.0.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL3_0_2_"
        x = conv2d(x, resultFilePath, resnet18.layer3[0].conv2)
        x = resnet18.get_submodule("layer3.0.bn2")(x)

        resultFilePath = "cL3_0_down_"
        identity = conv2d(identity, resultFilePath, resnet18.get_submodule("layer3.0.downsample.0"))
        identity = resnet18.get_submodule("layer3.0.downsample.1")(identity)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        identity = x

        resultFilePath = "cL3_1_1_"
        x = conv2d(x, resultFilePath, resnet18.layer3[1].conv1)
        x = resnet18.get_submodule("layer3.1.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL3_1_2_"
        x = conv2d(x, resultFilePath, resnet18.layer3[1].conv2)
        x = resnet18.get_submodule("layer3.1.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        # LAYER 4

        identity = x

        resultFilePath = "cL4_0_1_"
        x = conv2d(x, resultFilePath, resnet18.layer4[0].conv1)
        x = resnet18.get_submodule("layer4.0.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL4_0_2_"
        x = conv2d(x, resultFilePath, resnet18.layer4[0].conv2)
        x = resnet18.get_submodule("layer4.0.bn2")(x)

        resultFilePath = "cL4_0_down_"
        identity = conv2d(identity, resultFilePath, resnet18.get_submodule("layer4.0.downsample.0"))
        identity = resnet18.get_submodule("layer4.0.downsample.1")(identity)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        identity = x

        resultFilePath = "cL4_1_1_"
        x = conv2d(x, resultFilePath, resnet18.layer4[1].conv1)
        x = resnet18.get_submodule("layer4.1.bn1")(x)
        x = nn.ReLU(inplace=True)(x)

        resultFilePath = "cL4_1_2_"
        x = conv2d(x, resultFilePath, resnet18.layer4[1].conv2)
        x = resnet18.get_submodule("layer4.1.bn2")(x)

        x += identity
        x = nn.ReLU(inplace=True)(x)

        # CATEGORIZE

        x = resnet18.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = resnet18.fc(x)
        
        return x, control 

 
    def preprocess(self, image):
        return M.ResNet18_Weights.IMAGENET1K_V1.transforms()(image) 
        
        
    def __repr__(self): 
        return repr(self.model) 


class vgg16:
    def __init__(self): 
        self.model = M.vgg16(weights=M.VGG16_Weights.IMAGENET1K_V1).eval()
        
    
    # Returns (x, control) 
    def __call__(self, dataset):
        vgg16 = self.model 
        x = dataset 
        control = vgg16(x.clone())
        
        # PART 1
        
        resultFilePath = "cL1_1_"
        x = conv2d(x, resultFilePath, vgg16.features[0])  # Conv2D-1
        x = vgg16.features[1](x)                          # ReLU-2
    
        resultFilePath = "cL1_2_"
        x = conv2d(x, resultFilePath, vgg16.features[2])  # Conv2D-3
        x = vgg16.features[3](x)                          # ReLU-4
        x = vgg16.features[4](x)                          # MaxPool2D-5
        
        # PART 2 
        
        resultFilePath = "cL2_1_"
        x = conv2d(x, resultFilePath, vgg16.features[5])  # Conv2D-6
        x = vgg16.features[6](x)                          # ReLU-7
        
        resultFilePath = "cL2_2_"
        x = conv2d(x, resultFilePath, vgg16.features[7])  # Conv2D-8
        x = vgg16.features[8](x)                          # ReLU-9
        x = vgg16.features[9](x)                          # MaxPool2D-10
        
        # PART 3 
        
        resultFilePath = "cL3_1_"
        x = conv2d(x, resultFilePath, vgg16.features[10]) # Conv2D-11
        x = vgg16.features[11](x)                         # ReLU-12
        
        resultFilePath = "cL3_2_"
        x = conv2d(x, resultFilePath, vgg16.features[12]) # Conv2D-13
        x = vgg16.features[13](x)                         # ReLU-14
        
        resultFilePath = "cL3_3_"
        x = conv2d(x, resultFilePath, vgg16.features[14]) # Conv2D-15
        x = vgg16.features[15](x)                         # ReLU-16
        x = vgg16.features[16](x)                         # MaxPool2D-17
        
        # PART 4
        
        resultFilePath = "cL4_1_"
        x = conv2d(x, resultFilePath, vgg16.features[17]) # Conv2D-18
        x = vgg16.features[18](x)                         # ReLU-19
        
        resultFilePath = "cL4_2_"
        x = conv2d(x, resultFilePath, vgg16.features[19]) # Conv2D-20
        x = vgg16.features[20](x)                         # ReLU-21
        
        resultFilePath = "cL4_3_"
        x = conv2d(x, resultFilePath, vgg16.features[21]) # Conv2D-22
        x = vgg16.features[22](x)                         # ReLU-23
        x = vgg16.features[23](x)                         # MaxPool2D-24
        
        # PART 5

        resultFilePath = "cL5_1_"
        x = conv2d(x, resultFilePath, vgg16.features[24]) # Conv2D-25
        x = vgg16.features[25](x)                         # ReLU-26
        
        resultFilePath = "cL5_2_"
        x = conv2d(x, resultFilePath, vgg16.features[26]) # Conv2D-27
        x = vgg16.features[27](x)                         # ReLU-28
        
        resultFilePath = "cL5_3_"
        x = conv2d(x, resultFilePath, vgg16.features[28]) # Conv2D-29
        x = vgg16.features[29](x)                         # ReLU-30
        x = vgg16.features[30](x)                         # MaxPool2D-31

        # CATEGORIZE
        
        x = vgg16.avgpool(x) 
        x = torch.flatten(x, start_dim=1) 
        x = vgg16.classifier(x) # linear, relu, linear, relu, linear
        
        return x, control 

 
    def preprocess(self, image):
        return M.VGG16_Weights.IMAGENET1K_V1.transforms()(image) 
        
        
    def __repr__(self): 
        return repr(self.model) 
        
        
class alexnet:
    def __init__(self): 
        self.model = M.alexnet(weights=M.AlexNet_Weights.IMAGENET1K_V1).eval()
        
    
    # Returns (x, control) 
    def __call__(self, dataset):
        alexnet = self.model 
        x = dataset 
        control = alexnet(x.clone())
        
        # PART 1
        
        resultFilePath = "cL1_"
        x = conv2d(x, resultFilePath, alexnet.features[0])  # Conv2D-1
        x = alexnet.features[1](x)                          # ReLU-2
        x = alexnet.features[2](x)                          # MaxPool2D-3 
    
        # PART 2
        
        resultFilePath = "cL2_"
        x = conv2d(x, resultFilePath, alexnet.features[3])  # Conv2D-4
        x = alexnet.features[4](x)                          # ReLU-5
        x = alexnet.features[5](x)                          # MaxPool2D-6 
        
        # PART 3
        
        resultFilePath = "cL3_"
        x = conv2d(x, resultFilePath, alexnet.features[6])  # Conv2D-7
        x = alexnet.features[7](x)                          # ReLU-8
        
        # PART 4
        
        resultFilePath = "cL4_"
        x = conv2d(x, resultFilePath, alexnet.features[8])  # Conv2D-9
        x = alexnet.features[9](x)                          # ReLU-10

        # PART 5
        
        resultFilePath = "cL5_"
        x = conv2d(x, resultFilePath, alexnet.features[10]) # Conv2D-11
        x = alexnet.features[11](x)                         # ReLU-12
        x = alexnet.features[12](x)                         # MaxPool2D-13

        # CATEGORIZE
        
        x = alexnet.avgpool(x) 
        x = torch.flatten(x, start_dim=1) 
        x = alexnet.classifier(x) # linear, relu, linear, relu, linear
        
        return x, control 

 
    def preprocess(self, image):
        return M.AlexNet_Weights.IMAGENET1K_V1.transforms()(image) 
        
        
    def __repr__(self): 
        return repr(self.model) 
        
        
class yolo:
    def __init__(self): 
        self.model = torch.hub.load(
            'ultralytics/yolov5', 
            'custom', 
            'models/yolov5l-cls.pt'
        )
        
        # "self.model" alone is a wrapper, so we need to extract the part that's 
        # actually relevant to us. 
        self.model = self.model.model.model.to('cpu').eval()
        
    
    # Returns (x, control) 
    def __call__(self, dataset):
        yolo = self.model
        x = dataset 
        control = yolo(x.clone()) 
        
        # Layer implementations are taken from: 
        #   https://github.com/ultralytics/yolov5/blob/master/models/common.py
        
        def conv(x, layer, name): 
            x = conv2d(x, name, layer.conv) 
            x = layer.act(x) 
            return x 
            
        def bottleneck(x, layer, name): 
            y = conv(x, layer.cv1, name + '0_')
            z = conv(y, layer.cv2, name + '1_') 
            return x + z 
        
        def c3(x, layer, name): 
            a = conv(x, layer.cv1, name + '0_')
            b = a 
            for i in range(len(layer.m)):
                b = bottleneck(b, layer.m[i], f'{name}1_{i}_')
            c = conv(x, layer.cv2, name + '2_') 
            d = torch.cat((b, c), 1) 
            e = conv(d, layer.cv3, name + '3_') 
            return e
        
        def classify(x, layer, name): 
            # (There's only one classifier layer, but we can implement it like 
            # this for the sake of consistency)
            x = conv(x, layer.conv, name) 
            x = layer.pool(x)
            x = x.flatten(1) 
            x = layer.linear(x) 
            return x 
        
        layers = [conv, conv, c3, conv, c3, conv, c3, conv, c3, classify] 
        for i in range(len(layers)):
            x = layers[i](x, yolo[i], f'cL{i}_')
        
        return x, control 

 
    def preprocess(self, image):
        return M.AlexNet_Weights.IMAGENET1K_V1.transforms()(image) 
        
        
    def __repr__(self): 
        return repr(self.model) 


def save_matrix(x, fpath):
    assert x.dim() == 2

    with open(fpath, "wb") as f:
        f.write(x.size(dim=0).to_bytes(4, byteorder="little"))
        f.write(x.size(dim=1).to_bytes(4, byteorder="little"))
        f.write(x.numpy().tobytes())


def load_matrix(fpath) -> torch.Tensor:
    with open(fpath, "rb") as f:
        h = int.from_bytes(f.read(4), byteorder="little")
        w = int.from_bytes(f.read(4), byteorder="little")

        buf = f.read(h * w * np.dtype(np.float32).itemsize)

        x = torch.from_numpy(np.frombuffer(buf, dtype=np.float32).copy())
        x = x.reshape(h, w)

        return x


def pad_matrix(x) -> torch.Tensor:
    assert x.dim() == 2

    h = x.size(dim=0)
    w = x.size(dim=1)

    if h % 16 != 0:
        padding = 16 - h % 16
        x = F.pad(x, pad=(0, 0, 0, padding), mode="constant", value=0.0)

    if w % 16 != 0:
        padding = 16 - w % 16
        x = F.pad(x, pad=(0, padding, 0, 0), mode="constant", value=0.0)

    return x


def conv2d(x, filePath, module) -> torch.Tensor:
    weight = module.weight.detach()
    x = x.detach()

    out_n = x.size(dim=0)
    out_c = weight.size(dim=0)
    out_h = math.floor(((x.size(dim=2) + 2 * module.padding[0] - module.dilation[0] * (
        module.kernel_size[0] - 1) - 1) / module.stride[0]) + 1)
    out_w = math.floor(((x.size(dim=3) + 2 * module.padding[1] - module.dilation[1] * (
        module.kernel_size[1] - 1) - 1) / module.stride[1]) + 1)

    weight = weight.flatten(start_dim=1)
    weight = weight.view(weight.size(dim=0), weight.size(dim=1))

    x = nn.Unfold(kernel_size=module.kernel_size, stride=module.stride,
                  padding=module.padding, dilation=module.dilation)(x)

    slices = []
    for i in range(x.size(dim=0)):
        slices.append(x[i])
    x = torch.cat(slices, dim=1)

    h = weight.size(dim=0)
    w = x.size(dim=1)

    weight = pad_matrix(weight)
    x = pad_matrix(x)

    weightFilePath = "bin/weight_" + filePath + ".bin"
    xFilePath = "bin/x_" + filePath + ".bin"
    save_matrix(weight, weightFilePath)
    save_matrix(x, xFilePath)

    outputFilePath = "simResults_" + filePath + ".txt"
    errFilePath = "simErrors_" + filePath + ".txt"
    outputFile = open(outputFilePath, "w")
    errFile = open(errFilePath, "w")

    result = subprocess.run(["./build/gemm", 
                                "--w", 
                                weightFilePath,
                                "--x", 
                                xFilePath], 
                            stdout=outputFile, 
                            stderr=errFile)
    # subprocess.STDOUT

    print(result)

    x = load_matrix("bin/gemm.bin")
    gemmFilePath = "bin/gemm_" + filePath + ".bin"
    save_matrix(x, gemmFilePath) # save for tracing

    x = torch.stack(torch.chunk(x[:h, :w], chunks=out_n, dim=1))
    x = x.view(out_n, out_c, out_h, out_w)

    for name, _ in module.named_parameters():
        if name in ["bias"]:
            bias = module.bias.detach()
            bias = bias.view(1, bias.size(dim=0), 1, 1)
            bias = bias.tile(1, 1, out_h, out_w)

            x = x.add(bias)

    return x


if __name__ == "__main__":
    
    # python3 run.py <model type>
    #   <model type> is a string, can possibly start with a dash "-", and must 
    #   be either "resnet18"/"resnet", "vgg16"/"vgg", "alexnet", or 
    #   "yolov5l"/"yolo"
    
    name = sys.argv[-1].lower()
    while name.startswith('-'): name = name[1:] 
    
    if name == 'resnet18' or name == 'resnet':
        model = resnet18() 
    elif name == 'vgg16' or name == 'vgg': 
        model = vgg16() 
    elif name == 'alexnet':
        model = alexnet()
    elif name == 'yolov5l' or name == 'yolo': 
        model = yolo() 
    else:
        out = 'Unrecognized model name: pass either "-resnet18", "-vgg16", ' \
              'or "-alexnet" as an argument.'
        print(out) 
        sys.exit(1) 
    
    print(model) 
    
    # Images are stored within the "data" directory.
    filenames = []
    images    = []  
    if os.path.exists('data'): 
        pathnames = glob(os.path.join("data", "*.jpg"))
        for file in sorted(pathnames, key=os.path.basename):
            # Each model has their own unique "preprocess" method. 
            tensor = model.preprocess(Image.open(file))
            name = file[5:] # removes the 'data/' part 
            filenames.append(name) 
            images.append(tensor) 
    
    if len(images) == 0:
        out = 'Error: at least one .jpg image must be stored within the ' \
              '"data" directory.'
        print(out) 
        sys.exit(1) 
        
    # Output bin files are stored within the bin directory. 
    if not os.path.exists('bin'): 
        os.mkdir('bin') 
    
    # Models are callable: this runs an inference on the images. 
    dataset = torch.stack(images)
    x, control = model(dataset) 
    
    print(f'\nMSE: {nn.MSELoss()(x, control).item()}\n')
    
    # Added so output values represent confidence/probabilities
    x = F.softmax(x, dim=1)
    
    # Formatting 
    longest_filename = max([len(name) for name in filenames]) + 3 # padding 
    format_str = '%-' + str(longest_filename) + 's: %s'
    
    print("Classifications:")
    for index in range(len(images)):
        path = filenames[index] 
        argmax = torch.argmax(x[index]).item() 
        label = label_to_name(argmax)
        confidence = x[index][argmax].item() 
        
        padding = '.' * (longest_filename - len(path))
        confidence_str = '[%.3f] ' % confidence 
        print(path + padding + confidence_str + label) 