import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))


class TraBlock1(nn.Module):
    def __init__(self, dim):
        super(TraBlock1, self).__init__()

        self.res1 = ResBlock(dim)

    def forward(self, x):
        x = self.res1(x)

        return x


class TraBlock2(nn.Module):
    def __init__(self, dim):
        super(TraBlock2, self).__init__()

        self.dowm = nn.Conv2d(dim, dim * 2, kernel_size=3, padding=1, stride=2, bias=True)
        self.res1 = ResBlock(dim * 2)

    def forward(self, x):
        x = self.res1(self.dowm(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        return x


class TraBlock(nn.Module):
    def __init__(self, dim):
        super(TraBlock, self).__init__()

        self.tra1 = TraBlock1(dim)
        self.tra2 = TraBlock2(dim)
        self.conv1 = nn.Conv2d(dim * 3, dim, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x):
        x1 = self.tra1(x)
        x2 = self.tra2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv1(x)

        return x


class GradBlock(nn.Module):
    def __init__(self, dim):
        super(GradBlock, self).__init__()

        self.conv1 = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)
        self.res1 = ResBlock(dim)

    def forward(self, x, y, Phi, PhiT):
        x_pixel = self.conv1(x)
        Phix = F.conv2d(x_pixel, Phi, padding=0, stride=32, bias=None)
        delta = y - Phix
        x_pixel = nn.PixelShuffle(32)(F.conv2d(delta, PhiT, padding=0, bias=None))
        x_delta = self.conv2(x_pixel)
        x = self.res1(x_delta) + x

        return x


class FSOINet(nn.Module):
    def __init__(self, sensing_rate, LayerNo):
        super(FSOINet, self).__init__()

        self.measurement = int(sensing_rate * 1024)
        self.base = 16

        self.Phi = nn.Parameter(init.xavier_normal_(torch.Tensor(self.measurement, 1024)))
        self.conv1 = nn.Conv2d(1, self.base, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(self.base, 1, kernel_size=1, padding=0, bias=True)
        layer1 = []
        layer2 = []
        self.LayerNo = LayerNo
        for i in range(LayerNo):
            layer1.append(TraBlock(self.base))
            layer2.append(GradBlock(self.base))
        self.fcs1 = nn.ModuleList(layer1)
        self.fcs2 = nn.ModuleList(layer2)

    def forward(self, x):
        Phi = self.Phi.contiguous().view(self.measurement, 1, 32, 32)
        PhiT = self.Phi.t().contiguous().view(1024, self.measurement, 1, 1)
        y = F.conv2d(x, Phi, padding=0, stride=32, bias=None)
        x = F.conv2d(y, PhiT, padding=0, bias=None)
        x = nn.PixelShuffle(32)(x)

        x = self.conv1(x)
        for i in range(self.LayerNo):
            x = self.fcs2[i](x, y, Phi, PhiT)
            x = self.fcs1[i](x)
        x = self.conv2(x)

        phi_cons = torch.mm(self.Phi, self.Phi.t()).squeeze().squeeze()

        return x, phi_cons
