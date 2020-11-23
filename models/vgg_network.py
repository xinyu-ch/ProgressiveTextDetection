import torch.nn as nn
import torch
import torch.nn.functional as F
from models.pool_direction import make_pool_layer


class Base_with_bn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, up=False):
        super(Base_with_bn_block, self).__init__()
        self.up = up
        if up:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int(kernel_size / 2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self._initialize_weights()

    def forward(self, x):
        if self.up:
            x = self.up(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Base_down_block(nn.Module):
    def __init__(self, in_channels, out_channels, times):
        super(Base_down_block, self).__init__()
        # self.decon = ConvOffset2D(filters=in_channels)
        self.blocks = [Base_with_bn_block(in_channels, out_channels, 3)]
        for i in range(times - 1):
            self.blocks += [Base_with_bn_block(out_channels, out_channels, 3)]
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        # x = self.decon(x)
        out = self.blocks(x)
        return out


class Base_up_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Base_up_block, self).__init__()
        self.block1 = Base_with_bn_block(in_channels, out_channels * 2, 1, up=True)
        self.block2 = Base_with_bn_block(out_channels * 2, out_channels, 3)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')

    def forward(self, x1, x2):
        x2 = self._upsample_add(x2, x1)
        out = torch.cat([x1, x2], 1)
        out = self.block1(out)
        out = self.block2(out)
        return out


class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.layers = nn.Sequential(*[Base_down_block(3, 64, 2),
                                      Base_down_block(64, 128, 2),
                                      Base_down_block(128, 256, 3),
                                      Base_down_block(256, 512, 3),
                                      Base_down_block(512, 512, 3),
                                      Base_down_block(512, 512, 3), ])

        self.up_layers = nn.Sequential(*[Base_up_block(512 + 512, 256),
                                         Base_up_block(512 + 256, 128),
                                         Base_up_block(256 + 128, 64),
                                         Base_up_block(128 + 64, 64)])

        self.detector = nn.Sequential(*[Base_with_bn_block(64, 32, 1, up=True),
                                        Base_with_bn_block(32, 16, 3),
                                        Base_with_bn_block(16, 16, 3)])

        self.pool_bridge = nn.Sequential(*[make_pool_layer(128),
                                           make_pool_layer(256),
                                           make_pool_layer(512),
                                           make_pool_layer(512),
                                           make_pool_layer(512)])

        self.center = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        def _upsample_add(x):
            _, _, H, W = size
            return F.upsample(x, size=(H, W), mode='bilinear')
        size = x.size()
        features = []
        for i in range(5):
            x = self.layers[i](x)
            x = self.pooling(x)
            if i > 0:
                features.append(self.pool_bridge[i-1](x))
        x = self.layers[-1](x)
        x = self.pool_bridge[-1](x)
        for index in range(4):
            x = self.up_layers[index](features[-index - 1], x)
        x = self.detector(x)
        x = _upsample_add(x)
        center = self.center(x)
        center = torch.sigmoid(center)
        return center
