import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ReconNetWrapper(nn.Module):
    def __init__(self, net_recon='resnet50', use_last_fc=False):
        super(ReconNetWrapper, self).__init__()
        self.use_last_fc = use_last_fc
        
        # ResNet50 backbone
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        last_dim = 2048  # ResNet50's output dimension
        if not use_last_fc:
            self.final_layers = nn.ModuleList([
                conv1x1(last_dim, 80, bias=True),  # id layer
                conv1x1(last_dim, 64, bias=True),  # exp layer
                conv1x1(last_dim, 80, bias=True),  # tex layer
                conv1x1(last_dim, 3, bias=True),   # angle layer
                conv1x1(last_dim, 27, bias=True),  # gamma layer
                conv1x1(last_dim, 2, bias=True),   # tx, ty
                conv1x1(last_dim, 1, bias=True)    # tz
            ])
            for m in self.final_layers:
                nn.init.constant_(m.weight, 0.)
                nn.init.constant_(m.bias, 0.)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        
        if not self.use_last_fc:
            output = []
            for layer in self.final_layers:
                output.append(layer(x))
            x = torch.cat(output, dim=1)
            x = x.view(x.size(0), -1)
        
        return {
            'shape': x[:, :80],       # shape coefficients
            'exp': x[:, 80:144],      # expression coefficients
            'tex': x[:, 144:224],     # texture coefficients
            'angle': x[:, 224:227],   # euler angles
            'gamma': x[:, 227:254],   # lighting parameters
            'trans': x[:, 254:257]    # translation
        }

    def decode_shape(self, shape_coeffs, exp_coeffs):
        """
        Decode shape and expression coefficients to get vertex positions
        This is a simplified version - in practice you should use BFM model
        """
        # Simplified: just combine shape and expression
        return shape_coeffs + exp_coeffs
