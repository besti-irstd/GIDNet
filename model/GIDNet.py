import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.fft
import numpy as np

# 设置 CUDA 设备 (根据需要修改)，本文采用单GPU RTX 4090，其余配置见论文中有详细阐述
os.environ['CUDA_VISIBLE_DEVICES']="0"

# ==========================================
# 1. 基础卷积与注意力模块 (CDC, Attention)
# ==========================================

class GISC(nn.Module):#这里并没有启动偏置，
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, theta=0.7):
        super(GISC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        
        # 权重初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        # 1. 普通卷积结果
        out_normal = self.conv(x)

        if self.theta == 0:
            return out_normal
        
        # 2. 计算中心差分卷积
        weight = self.conv.weight
        kernel_sum = weight.sum(dim=(2, 3), keepdim=True)
        
        # 创建中心掩码
        mask = torch.zeros_like(weight)
        mask[:, :, weight.shape[2]//2, weight.shape[3]//2] = 1
        
        weight_diff = weight - kernel_sum * mask
        
        # 3. 差分卷积结果
        out_diff = F.conv2d(input=x, weight=weight_diff, bias=self.conv.bias, 
                            stride=self.conv.stride, padding=self.conv.padding, 
                            dilation=self.conv.dilation, groups=self.conv.groups)

        # 4. 加权融合
        return out_normal * (1 - self.theta) + out_diff * self.theta

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, max(1, in_planes // 16), 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(max(1, in_planes // 16), in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# ==========================================
# 2. 骨干网络模块 (ResNet Block, MSDC)
# ==========================================

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out

# MSDC
class MSDC(nn.Module):
    def __init__(self, inplanes, outplanes, one, two, three, scales = 4):
        super(MSDC, self).__init__()
        if outplanes % scales != 0: 
            raise ValueError('Planes must be divisible by scales')
        
        self.scales = scales
        self.relu = nn.ReLU(inplace = True)
        self.spx = outplanes // scales
        self.inconv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, 1, 1, 0),
            nn.BatchNorm2d(outplanes)
        )

        # 使用 CDC_Conv 替换普通卷积以增强边缘提取
        self.conv1 = nn.Sequential(
            GISC(self.spx, self.spx, kernel_size=one, stride=1, padding=one // 2, groups=self.spx, dilation=1),
            nn.BatchNorm2d(self.spx),
        )
        self.conv2 = nn.Sequential(
            GISC(self.spx, self.spx, kernel_size=two, stride=1, padding=2, groups=self.spx, dilation=2),
            nn.BatchNorm2d(self.spx),
        )
        self.conv3 = nn.Sequential(
            GISC(self.spx, self.spx, kernel_size=three, stride=1, padding=1, groups=self.spx, dilation=1),
        )
        self.conv4 = nn.Sequential(
            GISC(self.spx, self.spx, kernel_size=three, stride=1, padding=2, groups=self.spx, dilation=2),
        )
        
        self.conv5 = nn.Sequential(
            nn.BatchNorm2d(self.spx)
        )
        self.outconv = nn.Sequential(
            nn.Conv2d(outplanes, outplanes, 3, 1, 1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.inconv(x)
        inputt = x
        xs = torch.chunk(x, self.scales, 1)
        ys = []
        
        # 多尺度级联处理
        ys.append(xs[0])
        ys.append(self.relu(self.conv1(xs[1])))
        ys.append(self.relu(self.conv2(xs[2] + ys[1])))
        
        temp = xs[3] + ys[2]
        temp1 = self.conv5(self.conv3(temp) + self.conv4(temp))
        ys.append(self.relu(temp1))
        
        y = torch.cat(ys, 1)
        y = self.outconv(y)
        output = self.relu(y + inputt)
        return output

# ==========================================
# 3. 频域处理模块 (DHPF)，在本文中并没有使用频域模块
# ==========================================

class DHPF(nn.Module):
    def __init__(self, energy):
        super(DHPF, self).__init__()
        self.energy = energy
    


# ==========================================
# 4. GIDNet 主网络 (集成 Path Fusion Strategy)
# ==========================================


class GIDNet(nn.Module):
    def __init__(self, input_channels, block=ResNet):
        super(GIDNet, self).__init__()
        # 通道与层数配置
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        energy = [0.1, 0.2, 0.4, 0.8]

        # 上采样与池化
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        
        # 初始层
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)
        self.py_init = self._make_layer2(input_channels, 1, block)

        # ----------------- Encoder (Spatial Domain) -----------------
        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])
     
        # Middle Layer
        self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])
        
        # ----------------- Decoder (Spatial Domain) -----------------
        self.decoder_3 = self._make_layer2(param_channels[3]+param_channels[4], param_channels[3], block, param_blocks[2])
        self.decoder_2 = self._make_layer2(param_channels[2]+param_channels[3], param_channels[2], block, param_blocks[1])
        self.decoder_1 = self._make_layer2(param_channels[1]+param_channels[2], param_channels[1], block, param_blocks[0])
        self.decoder_0 = self._make_layer2(param_channels[0]+param_channels[1], param_channels[0], block)

        # [ Path Fusion] ----SFP策略
        self.path_fusion = nn.Sequential(
            nn.Conv2d(param_channels[0], param_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(param_channels[0]),
            nn.ReLU(inplace=True)
        )

        # ----------------- Frequency Domain Modules -----------------（会有控制器控制频域模块并不参与）
        self.py3 = DHPF(energy[3])
        self.py2 = DHPF(energy[2])
        self.py1 = DHPF(energy[1])
        self.py0 = DHPF(energy[0])

        # Prediction Heads
        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)
        self.output_1 = nn.Conv2d(param_channels[1], 1, 1)
        self.output_2 = nn.Conv2d(param_channels[2], 1, 1)
        self.output_3 = nn.Conv2d(param_channels[3], 1, 1)

        self.final = nn.Conv2d(4, 1, 3, 1, 1)

    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        # 使用 MSDC
        layer.append(MSDC(in_channels, out_channels, 3, 3, 3))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)
    
    def _make_layer2(self, in_channels, out_channels, block, block_num = 1):
        layer= []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x, warm_flag):
        
        # 1. Forward Pass - Encoder
        x_e0 = self.encoder_0(self.conv_init(x)) 
        x_e1 = self.encoder_1(self.pool(x_e0))
        x_e2 = self.encoder_2(self.pool(x_e1))
        x_e3 = self.encoder_3(self.pool(x_e2))

        # 2. Forward Pass - Middle
        x_m = self.middle_layer(self.pool(x_e3))
        
        # 3. Forward Pass - Decoder
        x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1)) 
        
        # [执行策略: Path Fusion] 
        x_d0 = x_d0 + self.path_fusion(x_e0)

        # 4. Generate Multi-scale Masks
        mask0 = self.output_0(x_d0)
        mask1 = self.output_1(x_d1)
        mask2 = self.output_2(x_d2)
        mask3 = self.output_3(x_d3)
        
        if warm_flag:
            print("")
            # 频域分支处理 (Frequency Domain Branch)，这里完全就没有频域处理逻辑
        else:#GIDNet不使用频域分支是本论文采用的方法，也没有深度loss监督
            output = self.output_0(x_d0)
            return [], output
