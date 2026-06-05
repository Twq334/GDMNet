import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import fvcore.nn.weight_init as weight_init
from model.GatedConv import GatedConv2dWithActivation
from kornia.filters import laplacian
from einops import rearrange
import numbers

from timm.models.layers import to_2tuple
from FDConv import FDConv
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Softmax, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()


def _get_act_fn(act_name, inplace=True):
    if act_name == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_name == "leaklyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=inplace)
    elif act_name == "gelu":
        return nn.GELU()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError

def resize_to(x: torch.Tensor, tgt_hw: tuple):
    return F.interpolate(x, size=tgt_hw, mode="bilinear", align_corners=False)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        act_name="relu",
        is_transposed=False,
    ):
        """
        Convolution-BatchNormalization-ActivationLayer

        :param in_planes:
        :param out_planes:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        :param act_name: None denote it doesn't use the activation layer.
        :param is_transposed: True -> nn.ConvTranspose2d, False -> nn.Conv2d
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        if is_transposed:
            conv_module = nn.ConvTranspose2d
        else:
            conv_module = nn.Conv2d
        self.add_module(
            name="conv",
            module=conv_module(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=to_2tuple(stride),
                padding=to_2tuple(padding),
                dilation=to_2tuple(dilation),
                groups=groups,
                bias=bias,
            ),
        )
        self.add_module(name="bn", module=nn.BatchNorm2d(out_planes))
        if act_name is not None:
            self.add_module(name=act_name, module=_get_act_fn(act_name=act_name))


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

    def initialize(self):
        weight_init(self)


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    def initialize(self):
        weight_init(self)



class MultiScaleFFN(nn.Module):
    def __init__(self, channels, expansion_factor, use_bias):
        super(MultiScaleFFN, self).__init__()
        hidden_dim = int(channels * expansion_factor)
        self.input_proj = nn.Conv2d(channels, hidden_dim * 2, kernel_size=1, bias=use_bias)
        self.dwconv_3x3 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_dim * 2, bias=use_bias)
        self.dwconv_5x5 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_dim * 2, bias=use_bias)
        self.dwconv_7x7 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_dim * 2, bias=use_bias)

        self.output_proj = nn.Conv2d(hidden_dim * 3, channels, kernel_size=1, bias=use_bias)

    def forward(self, x):
        x = self.input_proj(x)
        x1_3, x2_3 = self.dwconv_3x3(x).chunk(2, dim=1)
        x_3x3 = F.gelu(x1_3) * x2_3
        x1_5, x2_5 = self.dwconv_5x5(x).chunk(2, dim=1)
        x_5x5 = F.gelu(x1_5) * x2_5
        x1_7, x2_7 = self.dwconv_7x7(x).chunk(2, dim=1)
        x_7x7 = F.gelu(x1_7) * x2_7

        x = self.output_proj(torch.cat((x_3x3, x_5x5, x_7x7), 1))
        return x

    def initialize(self):
        weight_init(self)



class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, mode):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv_0 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_1 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.qkv_2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.qkv1conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv2conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.qkv3conv_3 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.qkv1conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv2conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)
        self.qkv3conv_5 = nn.Conv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim, bias=bias)

        self.qkv1conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv2conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)
        self.qkv3conv_7 = nn.Conv2d(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim*3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q_3 = self.qkv1conv_3(self.qkv_0(x))
        k_3 = self.qkv2conv_3(self.qkv_1(x))
        v_3 = self.qkv3conv_3(self.qkv_2(x))

        q_3 = rearrange(q_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_3 = rearrange(k_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_3 = rearrange(v_3, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_3 = torch.nn.functional.normalize(q_3, dim=-1)
        k_3 = torch.nn.functional.normalize(k_3, dim=-1)
        attn_3 = (q_3 @ k_3.transpose(-2, -1)) * self.temperature
        attn_3 = attn_3.softmax(dim=-1)
        out_3 = (attn_3 @ v_3)
        out_3 = rearrange(out_3, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_5 = self.qkv1conv_5(self.qkv_0(x))
        k_5 = self.qkv2conv_5(self.qkv_1(x))
        v_5 = self.qkv3conv_5(self.qkv_2(x))

        q_5 = rearrange(q_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_5 = rearrange(k_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_5 = rearrange(v_5, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_5 = torch.nn.functional.normalize(q_5, dim=-1)
        k_5 = torch.nn.functional.normalize(k_5, dim=-1)
        attn_5 = (q_5 @ k_5.transpose(-2, -1)) * self.temperature
        attn_5 = attn_5.softmax(dim=-1)
        out_5 = (attn_5 @ v_5)
        out_5 = rearrange(out_5, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        q_7 = self.qkv1conv_7(self.qkv_0(x))
        k_7 = self.qkv2conv_7(self.qkv_1(x))
        v_7 = self.qkv3conv_7(self.qkv_2(x))

        q_7 = rearrange(q_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_7 = rearrange(k_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_7 = rearrange(v_7, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_7 = torch.nn.functional.normalize(q_7, dim=-1)
        k_7 = torch.nn.functional.normalize(k_7, dim=-1)
        attn_7 = (q_7 @ k_7.transpose(-2, -1)) * self.temperature
        attn_7 = attn_7.softmax(dim=-1)
        out_7 = (attn_7 @ v_7)
        out_7 = rearrange(out_7, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(torch.cat((out_3,out_5,out_7),1))
        return out



    def initialize(self):
        weight_init(self)

class GroupFusionBlock(nn.Module):
    def __init__(self, channels):
        super(GroupFusionBlock, self).__init__()
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.freq_mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        spatial_out = self.conv_spatial(x)

        x_fft = torch.fft.fft2(x, norm="ortho")  # [B, C, H, W]
        x_fft_flat = x_fft.view(B, C, -1)
        amp = torch.abs(x_fft_flat)  #  [B, C, H*W]
        amp_trans = amp.transpose(1, 2)  # [B, H*W, C]
        attn = self.freq_mlp(amp_trans)  # [B, H*W, C]
        attn = torch.sigmoid(attn)
        attn = attn.transpose(1, 2).view(B, C, H, W)
        x_fft_mod = x_fft * attn.type_as(x_fft)
        freq_out = torch.fft.ifft2(x_fft_mod, norm="ortho").real

        out = spatial_out + self.gamma * freq_out
        return out


class MSA_head(Module):  # Multi-scale transformer block
    def __init__(self, mode='dilation', dim=128, num_heads=8, ffn_expansion_factor=4, bias=False,
                 LayerNorm_type='WithBias'):
        super(MSA_head, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias, mode)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = MultiScaleFFN(dim, ffn_expansion_factor, bias)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x
class NonLocal(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel

        # non-local
        temp_c = out_channel // 4
        self.query_conv = nn.Conv2d(in_channels=out_channel, out_channels=temp_c, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=out_channel, out_channels=temp_c, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # residual connection
        self.conv_res = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        # non-local
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        out = F.relu(x + self.conv_res(out1), inplace=False)
        return out

def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=False)
def getBasicBranch(in_channel, tmp_channel, out_channel, pool_kernel):
    conv1 = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=False)
    )
    pool1 = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel)
    conv3 = nn.Sequential(
        nn.Conv2d(tmp_channel, out_channel, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=False)
    )
    dilated = nn.Sequential(
        nn.Conv2d(out_channel, out_channel, 3, 1, 2, dilation=2, bias=False),
        nn.BatchNorm2d(out_channel),
        # nn.ReLU(inplace=False)
    )
    return conv1, pool1, conv3, dilated
#多尺度全局模块
# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享MLP
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # 平均池化路径
        max_out = self.fc(self.max_pool(x))  # 最大池化路径
        out = avg_out + max_out  # 融合两种池化结果
        return self.sigmoid(out)  # 通道注意力权重


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 通道平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 通道最大池化
        x_cat = torch.cat([avg_out, max_out], dim=1)  # 拼接特征
        out = self.conv(x_cat)  # 卷积生成空间特征
        return self.sigmoid(out)  # 空间注意力权重
# CBAM模块（通道注意力 + 空间注意力）
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先应用通道注意力，再应用空间注意力
        x = x * self.channel_att(x)  # 通道注意力加权
        x = x * self.spatial_att(x)  # 空间注意力加权
        return x


class MSGM(nn.Module):
    def __init__(self, in_channel, out_channel, nl=False):
        super().__init__()
        self.in_channel, self.out_channel = in_channel, out_channel
        self.nl = nl
        if self.nl:
            self.non_local_compress = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=False)
            )
            self.non_local = NonLocal(in_channel, out_channel)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=False)
        )
        self.conv1_branch1, self.pool1_branch1, self.conv3_branch1, self.dilated_branch1 = \
            getBasicBranch(in_channel, out_channel * 2, out_channel, 1)
        self.conv1_branch2, self.pool1_branch2, self.conv3_branch2, self.dilated_branch2 = \
            getBasicBranch(in_channel, out_channel * 2, out_channel, 2)
        self.conv1_branch3, self.pool1_branch3, self.conv3_branch3, self.dilated_branch3 = \
            getBasicBranch(in_channel, out_channel * 2, out_channel, 4)
        self.conv1_branch4, self.pool1_branch4, self.conv3_branch4, self.dilated_branch4 = \
            getBasicBranch(in_channel, out_channel, out_channel, 8)

    def forward(self, x):
        # nonlocal part
        if self.nl:
            x_nonlocal = self.non_local(self.non_local_compress(x))

        # multi-scale part
        # begin
        x_shortcut = self.shortcut(x)
        x1 = self.conv1_branch1(x)
        x2 = self.conv1_branch2(x)
        x3 = self.conv1_branch3(x)
        x4 = self.conv1_branch4(x)

        # merge
        x4 = self.dilated_branch4(self.conv3_branch4(self.pool1_branch4(x4)))
        x3 = self.dilated_branch3(self.conv3_branch3(self.pool1_branch3(
            torch.cat([x3, upsample(x4, x3.shape[2:])], dim=1)
        )))
        x2 = self.dilated_branch2(self.conv3_branch2(self.pool1_branch2(
            torch.cat([x2, upsample(x3, x2.shape[2:])], dim=1)
        )))
        x1 = self.dilated_branch1(self.conv3_branch1(
            torch.cat([x1, upsample(x2, x1.shape[2:])], dim=1)
        ))

        if self.nl:
            out = F.relu(x_shortcut + x1 + x_nonlocal, inplace=False)
        else:
            out = x_shortcut + x1
        return out#  #   #
#深度细节模块
class DDM(nn.Module):#以l4为例在模型中将GRD初始化为（512，128） x4（1，512，12，12）
    def __init__(self, in_channel, out_channel):
        super(DDM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),#(12+2*1-3)/1+1=12
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        #4 个分支同时对同一输入特征（x1）进行处理，输出 4 个尺寸相同（h, w 不变）但感受野不同的特征图，每个特征图侧重不同尺度的信息。
        self.aspp = nn.ModuleList([#存储子模块的列表容器，包含4个并行的卷积分支
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ) for rate in [1, 3, 5, 7]
        ])
        #调整aspp通道数
        self.reduce_aspp = nn.Sequential(
            nn.Conv2d(out_channel * 4, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.dw_conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dw_conv5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dw_conv7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.reduce_dw = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.se_fusion = SELayer(out_channel * 2)
        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 2, 3, padding=1),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel * 2, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        aspp_feats = [branch(x1) for branch in self.aspp]
        aspp_out = self.reduce_aspp(torch.cat(aspp_feats, dim=1))
        dw3 = self.dw_conv3(x1)
        dw5 = self.dw_conv5(x1)
        dw7 = self.dw_conv7(x1)
        dw_out = self.reduce_dw(torch.cat([dw3, dw5, dw7], dim=1))
        local_feat = self.se_fusion(torch.cat([aspp_out, dw_out], dim=1))
        res_feat = self.res(x)
        out = self.fuse_conv(torch.cat([local_feat, res_feat], dim=1)) + x0
        return out

class CoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 水平池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 垂直池化

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        # 水平池化
        x_h = self.pool_h(x)  # 形状: (n, c, h, 1)
        # 垂直池化
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # 形状: (n, c, 1, w) → 转置为 (n, c, w, 1)

        # 拼接并卷积
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 拆分
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # 转置回 (n, c, 1, w)

        # 生成注意力权重
        att_h = self.sigmoid(self.conv_h(x_h))
        att_w = self.sigmoid(self.conv_w(x_w))

        # 应用注意力
        return identity * att_h * att_w
class DDM_CA(nn.Module):  # 修改名称以体现使用CA
    def __init__(self, in_channel, out_channel):
        super(DDM_CA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        # ASPP分支
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channel, out_channel, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ) for rate in [1, 3, 5, 7]
        ])

        self.reduce_aspp = nn.Sequential(
            nn.Conv2d(out_channel * 4, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        # 深度可分离卷积分支
        self.dw_conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dw_conv5 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.dw_conv7 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1, groups=out_channel),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.reduce_dw = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        # 使用Coordinate Attention替换SELayer
        self.ca_fusion = CoordAtt(out_channel * 2)

        # 残差分支
        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 2, 3, padding=1),
            nn.BatchNorm2d(out_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel * 2, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(out_channel * 3, out_channel, 3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)

        # ASPP分支
        aspp_feats = [branch(x1) for branch in self.aspp]
        aspp_out = self.reduce_aspp(torch.cat(aspp_feats, dim=1))

        # 深度可分离卷积分支
        dw3 = self.dw_conv3(x1)
        dw5 = self.dw_conv5(x1)
        dw7 = self.dw_conv7(x1)
        dw_out = self.reduce_dw(torch.cat([dw3, dw5, dw7], dim=1))

        # 使用Coordinate Attention融合ASPP和DW分支的特征
        fused_feat = torch.cat([aspp_out, dw_out], dim=1)
        local_feat = self.ca_fusion(fused_feat)  # CA增强融合特征

        # 残差连接
        res_feat = self.res(x)

        # 最终融合
        out = self.fuse_conv(torch.cat([local_feat, res_feat], dim=1)) + x0
        return out
#特征梯度融合模块
class FGFM(nn.Module):
    def __init__(self, in_channels=128, groups=4):
        super(FGFM, self).__init__()
        self.in_channels = in_channels
        self.groups = groups
        group_channels = in_channels // groups

        # 原有宏微特征预处理
        self.pre_conv_G = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pre_conv_L = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 分组融合模块
        self.group_fusion = nn.ModuleList([
            GroupFusionBlock(2 * group_channels) for _ in range(groups)
        ])

        # 梯度注意力分支：输入为宏特征+梯度图，输出注意力权重
        self.gradient_attn = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels // 4, kernel_size=3, padding=1),  # 1为梯度图通道
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),  # 输出单通道权重图
            nn.Sigmoid()  # 权重归一化到[0,1]，边缘区域权重趋近1
        )

        # 特征降维与融合
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels * 4, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # 原有门控卷积与最终处理
        self.gated_conv = GatedConv2dWithActivation(
            in_channels, in_channels, kernel_size=3, stride=1,
            padding=1, dilation=1, groups=1, bias=True, batch_norm=True,
            activation=nn.LeakyReLU(0.2, inplace=True)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, G, L, x_ori=None):
        # 1. 宏微特征预处理
        G_feat = self.pre_conv_G(G)
        L_feat = self.pre_conv_L(L)

        # 2. 计算梯度图（输入原图x_ori为可选，若未提供则用G的均值近似）
        if x_ori is None:
            # 若未传入原图，用宏特征G的均值模拟图像（简化方案）
            x_gray = torch.mean(G, dim=1, keepdim=True)  # (B,1,H,W)
        else:
            # 用原图计算梯度（推荐，更准确）
            x_gray = torch.mean(x_ori, dim=1, keepdim=True)  # 原图转灰度
        grad_map = laplacian(x_gray, kernel_size=5)  # 拉普拉斯梯度图 (B,1,H,W)
        grad_map = F.interpolate(grad_map, size=G_feat.shape[-2:], mode='bilinear', align_corners=True)  # 适配特征尺寸

        # 3. 预测梯度注意力权重：基于宏特征和梯度图
        grad_attn = self.gradient_attn(torch.cat([G_feat, grad_map], dim=1))  # (B,1,H,W)

        # 4. 原有分组融合逻辑
        fusion_input = torch.cat([G_feat, L_feat], dim=1)
        groups = fusion_input.chunk(self.groups, dim=1)
        fused_groups = [self.group_fusion[i](groups[i]) for i in range(self.groups)]
        fused = torch.cat(fused_groups, dim=1)

        # 5. 梯度注意力加权：增强边缘区域特征
        fused = fused * grad_attn  # 用梯度权重加权融合特征

        # 6. 后续特征降维与处理
        concat_feat = torch.cat([fusion_input, fused], dim=1)
        out = self.reduce(concat_feat)
        out = self.gated_conv(out) * out + out
        out = self.final_conv(out)

        return out
#全局粗略解码模块
class GCDM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCDM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3,padding=1,dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*2, 3, 1, 1),nn.BatchNorm2d(out_channel*2),
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channel, out_channel//2, 3, padding=1),nn.BatchNorm2d(out_channel//2),nn.PReLU(), nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channel//2, 1, 1)
        )

        self.msa_head = MSA_head(dim=out_channel)



    def forward(self, x):

        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        x_multi = self.msa_head(x1)
        x_res  = self.res(x)
        x = self.reduce(torch.cat((x_res,x_multi),1)) + x0
        x = self.out(x)
        return x
class GRD(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GRD, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3,padding=1,dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*2, 3, 1, 1),nn.BatchNorm2d(out_channel*2),
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),
        )

        self.reduce  = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1),nn.BatchNorm2d(out_channel),nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channel, out_channel//2, 3, padding=1),nn.BatchNorm2d(out_channel//2),nn.PReLU(), nn.Dropout2d(p=0.1),
            nn.Conv2d(out_channel//2, 1, 1)
        )

        self.msa_head = MSA_head(dim=out_channel)



    def forward(self, x):

        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        x_multi = self.msa_head(x1)
        x_res  = self.res(x)
        x = self.reduce(torch.cat((x_res,x_multi),1)) + x0
        x = self.out(x)
        return x

class VPD_1(nn.Module): #Adjacent reverse decoder
    def __init__(self, in_channels, mid_channels):
        super(VPD_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )

        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )
        self.GL_FI = FGFM(in_channels)

    def forward(self, G, L, prior_cam):

        GL = self.GL_FI(G, L)

        prior_cam = F.interpolate(prior_cam, size=L.size()[2:], mode='bilinear', align_corners=True)

        yt = self.conv(torch.cat([GL, prior_cam.expand(-1, L.size()[1], -1, -1)], dim=1))

        conv_out = self.conv3(yt)

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        ra_out = r_prior_cam.expand(-1, L.size()[1], -1, -1).mul(GL)

        cat_out = torch.cat([ra_out, conv_out], dim=1)

        y = self.out_y(cat_out)

        y = y + prior_cam
        return y

class VPD_2(nn.Module): #  Adjacent reverse decoder
    def __init__(self, in_channels, mid_channels):
        super(VPD_2, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1), nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(in_channels), nn.ReLU(True),
        )

        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )

        self.GL_FI = FGFM(in_channels)

    def forward(self, G, L, x1, prior_cam):
        GL = self.GL_FI(G, L)

        prior_cam = F.interpolate(prior_cam, size=L.size()[2:], mode='bilinear',align_corners=True)  #
        x1_prior_cam = F.interpolate(x1, size=L.size()[2:], mode='bilinear', align_corners=True)

        yt = self.conv(torch.cat([GL, prior_cam.expand(-1, L.size()[1], -1, -1), x1_prior_cam.expand(-1, L.size()[1], -1, -1)],dim=1))
        conv_out = self.conv3(yt)

        r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1
        r1_prior_cam = -1 * (torch.sigmoid(x1_prior_cam)) + 1
        r_prior_cam = r_prior_cam + r1_prior_cam
        ra_out = r_prior_cam.expand(-1, L.size()[1], -1, -1).mul(GL)

        cat_out = torch.cat([ra_out, conv_out], dim=1)

        y = self.out_y(cat_out)
        y = y + prior_cam + x1_prior_cam
        return y


class GuidedReverseELA(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super(GuidedReverseELA, self).__init__()
        pad = kernel_size // 2

        # 对 prior 生成更平滑、更语义化的反向权重
        self.conv_prior = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        # ELA：沿 H/W 方向建模局部上下文
        self.conv1d = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                padding=pad, groups=channels, bias=False)
        self.gn = nn.GroupNorm(16, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prior):       # x: GL, prior: [B,1,H,W]
        b, c, h, w = x.size()
        # 先对 prior 做平滑，再得到反向权重
        p = self.conv_prior(prior)
        r = 1.0 - torch.sigmoid(p)      # 仍然是“反向”，但更平滑

        # 轴向统计 + 1D 卷积（ELA）
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        x_h = self.sigmoid(self.gn(self.conv1d(x_h))).view(b, c, h, 1)
        x_w = self.sigmoid(self.gn(self.conv1d(x_w))).view(b, c, 1, w)

        att = r * x_h * x_w             # 先验反向 × 轴向注意
        return x * att                  # 用这个注意力重新加权 GL
class VPD_1(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(VPD_1, self).__init__()
        # ... 你原来的 conv / conv3 / out_y 不变 ...
        self.GL_FI = FGFM(in_channels)
        self.rev_att = GuidedReverseELA(in_channels)   # 新加一行

    def forward(self, G, L, prior_cam):
        GL = self.GL_FI(G, L)
        prior_cam = F.interpolate(prior_cam, size=L.size()[2:],
                                  mode='bilinear', align_corners=True)

        yt = self.conv(torch.cat(
            [GL, prior_cam.expand(-1, L.size(1), -1, -1)], dim=1))

        conv_out = self.conv3(yt)

        # 原来的 r_prior_cam / ra_out 改成：
        ra_out = self.rev_att(GL, prior_cam)

        cat_out = torch.cat([ra_out, conv_out], dim=1)
        y = self.out_y(cat_out)
        y = y + prior_cam
        return y


# 从您提供的 VPD_1 代码中，我们得知需要以下基础模块，此处假设已定义
# from your_module import BasicConv2d, FGFM

class GuidedReverseELA_Fused(nn.Module):
    """
    方案二：融合先验引导的轴向反向注意力
    将两个先验图 (prior_cam 和 x1_prior_cam) 融合成一个综合信号，再生成引导式反向注意力。
    """

    def __init__(self, channels, kernel_size=7):
        super(GuidedReverseELA_Fused, self).__init__()
        pad = kernel_size // 2

        # 1. 融合两个先验图: [B,2,H,W] -> [B,1,H,W]
        self.fuse_priors = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        # 2. ELA：沿 H/W 方向建模局部上下文（与VPD_1中的模块相同）
        self.conv1d = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                                padding=pad, groups=channels, bias=False)
        self.gn = nn.GroupNorm(16, channels)  # 假设 channels >= 16
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, prior1, prior2):
        """
        Args:
            x (Tensor): 特征图 GL, 形状 [B, C, H, W]
            prior1 (Tensor): 第一个先验图 (prior_cam), 形状 [B, 1, H, W]
            prior2 (Tensor): 第二个先验图 (x1_prior_cam), 形状 [B, 1, H, W]
        Returns:
            Tensor: 经过引导式轴向反向注意力调制后的特征, 形状 [B, C, H, W]
        """
        b, c, h, w = x.size()

        # --- 步骤1: 融合双先验，并生成引导式反向权重 ---
        # 拼接两个先验图
        fused_prior = self.fuse_priors(torch.cat([prior1, prior2], dim=1))
        # 生成更平滑、更具语义的引导权重 (核心：可学习的“反向”)
        r = 1.0 - torch.sigmoid(fused_prior)  # r 形状: [B, 1, H, W]

        # --- 步骤2: 轴向注意力 (ELA) ---
        # 沿高度和宽度方向压缩，获取上下文
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        # 1D卷积处理轴向上下文
        x_h = self.sigmoid(self.gn(self.conv1d(x_h))).view(b, c, h, 1)
        x_w = self.sigmoid(self.gn(self.conv1d(x_w))).view(b, c, 1, w)

        # --- 步骤3: 融合引导权重与轴向注意力 ---
        att = r * x_h * x_w  # 调制后的注意力图，形状: [B, C, H, W]

        # --- 步骤4: 应用注意力，输出调制后的特征 ---
        return x * att


class VPD_2(nn.Module):
    """
    改进的相邻解码器 (VPD_2)
    使用 GuidedReverseELA_Fused 模块替换原有的静态反向注意力计算。
    """

    def __init__(self, in_channels, mid_channels):
        super(VPD_2, self).__init__()
        # 第一部分：特征融合与转换
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        # 第二部分：特征精炼
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )

        # 第三部分：输出预测
        self.out_y = nn.Sequential(
            BasicConv2d(in_channels * 2, mid_channels, kernel_size=3, padding=1),
            BasicConv2d(mid_channels, mid_channels // 2, kernel_size=3, padding=1),
            nn.Conv2d(mid_channels // 2, 1, kernel_size=3, padding=1)
        )

        # 第四部分：特征引导融合与新增的注意力模块
        self.GL_FI = FGFM(in_channels)  # 假设已定义
        self.rev_att = GuidedReverseELA_Fused(in_channels)  # 核心改进：融合引导注意力

    def forward(self, G, L, x1, prior_cam):
        """
        Args:
            G (Tensor): 全局特征
            L (Tensor): 局部特征
            x1 (Tensor): 来自 VPD_1 的输出预测，形状 [B, 1, H_prev, W_prev]
            prior_cam (Tensor): 初始先验预测，形状 [B, 1, H_prev, W_prev]
        Returns:
            y (Tensor): 本阶段的输出预测，形状 [B, 1, H, W] (H,W 同 L 的尺寸)
        """
        # 1. 融合全局与局部特征
        GL = self.GL_FI(G, L)

        # 2. 将两个先验图上采样到当前特征图 L 的尺寸
        prior_cam = F.interpolate(prior_cam, size=L.size()[2:],
                                  mode='bilinear', align_corners=True)
        x1_prior_cam = F.interpolate(x1, size=L.size()[2:],
                                     mode='bilinear', align_corners=True)

        # 3. 融合特征与先验信息，并进行初步转换
        yt = self.conv(torch.cat([
            GL,
            prior_cam.expand(-1, L.size()[1], -1, -1),
            x1_prior_cam.expand(-1, L.size()[1], -1, -1)
        ], dim=1))
        conv_out = self.conv3(yt)

        # 4. 【核心修改】使用融合引导式轴向反向注意力
        # 替换了原有的：r_prior_cam = -1 * (torch.sigmoid(prior_cam)) + 1 ...
        ra_out = self.rev_att(GL, prior_cam, x1_prior_cam)

        # 5. 拼接注意力输出与转换后的特征，生成最终预测
        cat_out = torch.cat([ra_out, conv_out], dim=1)
        y = self.out_y(cat_out)

        # 6. 残差连接，融合先验信息以稳定训练
        y = y + prior_cam + x1_prior_cam

        return y


class DynamicGRD(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DynamicGRD, self).__init__()
        # ... 其他层保持不变 ...

        # 替换为动态多尺度注意力头
        self.msa_head = DynamicMSA_head(dim=out_channel)

        # 添加尺度选择引导（可选）
        self.scale_guide = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 4, 3, padding=1),
            nn.BatchNorm2d(out_channel // 4),
            nn.ReLU(True),
            nn.Conv2d(out_channel // 4, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)

        # 生成尺度选择引导（告诉注意力哪些区域需要更精细的尺度）
        scale_guidance = self.scale_guide(x1)

        # 将引导信息传入动态注意力
        x_multi = self.msa_head(x1, scale_guidance)

        x_res = self.res(x)
        x = self.reduce(torch.cat((x_res, x_multi), 1)) + x0
        x = self.out(x)
        return x


class DynamicMSA_head(nn.Module):
    def __init__(self, dim=128, num_heads=8, ffn_expansion_factor=4, bias=False):
        super(DynamicMSA_head, self).__init__()
        self.norm1 = LayerNorm(dim)

        # 使用动态多尺度注意力
        self.attn = DynamicMultiScaleAttention(
            dim,
            num_heads=num_heads,
            bias=bias,
            num_scales=4  # 可以动态选择1-4个尺度
        )

        self.norm2 = LayerNorm(dim)
        self.ffn = MultiScaleFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x, guidance=None):
        b, c, h, w = x.shape
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 如果有引导信息，传入注意力模块
        if guidance is not None:
            x = x + self.attn(x_norm, guidance)
        else:
            x = x + self.attn(x_norm)

        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = x + self.ffn(x_norm)
        return x


class DynamicMultiScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False, num_scales=4):
        super(DynamicMultiScaleAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.head_dim = dim // num_heads

        # 1. 可选的尺度集合
        self.possible_scales = [1, 3, 5, 7, 9]  # 可选尺度

        # 2. 尺度选择器（新增）
        self.scale_selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, len(self.possible_scales), 1, bias=bias),
            nn.Softmax(dim=1)
        )

        # 3. 空间权重生成器（新增）
        self.spatial_weight_generator = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=bias),
            nn.GroupNorm(4, dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, len(self.possible_scales), 3, padding=1, bias=bias),
            nn.Softmax(dim=1)
        )

        # 4. 引导信息融合（新增）
        self.guidance_fusion = nn.Sequential(
            nn.Conv2d(1, dim // 8, 3, padding=1, bias=bias),  # 输入是引导图
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, len(self.possible_scales), 3, padding=1, bias=bias),
            nn.Tanh()  # 输出[-1,1]的调整因子
        )

        # 5. 多尺度QKV生成（类似原始）
        self.q_projs = nn.ModuleList()
        self.k_projs = nn.ModuleList()
        self.v_projs = nn.ModuleList()

        for scale in self.possible_scales:
            padding = scale // 2
            self.q_projs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, 1, bias=bias),
                    nn.Conv2d(dim, dim, scale, padding=padding, groups=dim, bias=bias)
                )
            )
            self.k_projs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, 1, bias=bias),
                    nn.Conv2d(dim, dim, scale, padding=padding, groups=dim, bias=bias)
                )
            )
            self.v_projs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, 1, bias=bias),
                    nn.Conv2d(dim, dim, scale, padding=padding, groups=dim, bias=bias)
                )
            )

        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x, guidance=None):
        b, c, h, w = x.shape

        # 步骤1: 生成基础融合权重
        base_weights = self.spatial_weight_generator(x)  # [B, num_scales, H, W]

        # 步骤2: 如果有引导信息，调整权重
        if guidance is not None:
            # 从引导信息生成调整因子
            adjustment = self.guidance_fusion(guidance)  # [B, num_scales, H, W]

            # 动态调整权重：在引导值高的区域增强某些尺度的作用
            # guidance通常是[0,1]范围，值越高表示该区域越复杂/重要
            final_weights = base_weights * (1 + adjustment * 0.5)
            final_weights = F.softmax(final_weights, dim=1)
        else:
            final_weights = base_weights

        # 步骤3: 计算各尺度注意力
        scale_outputs = []
        for i, scale in enumerate(self.possible_scales):
            # 生成当前尺度的QKV
            q_i = self.q_projs[i](x)
            k_i = self.k_projs[i](x)
            v_i = self.v_projs[i](x)

            # 计算注意力
            q_i = rearrange(q_i, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
            k_i = rearrange(k_i, 'b (head d) h w -> b head (h w) d', head=self.num_heads)
            v_i = rearrange(v_i, 'b (head d) h w -> b head (h w) d', head=self.num_heads)

            attn = torch.matmul(q_i, k_i.transpose(-2, -1)) * (self.head_dim ** -0.5)
            attn = attn.softmax(dim=-1)
            out_i = torch.matmul(attn, v_i)
            out_i = rearrange(out_i, 'b head (h w) d -> b (head d) h w', h=h, w=w)

            scale_outputs.append(out_i)

        # 步骤4: 动态加权融合（核心动态性）
        # 每个位置使用不同的尺度组合权重
        out = 0
        for i in range(len(self.possible_scales)):
            weight = final_weights[:, i:i + 1, :, :]  # [B, 1, H, W]
            out = out + scale_outputs[i] * weight

        out = self.project_out(out)
        return out