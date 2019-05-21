# implementation of our model:Weighted Residual Attention Network for Super Resolution

from model import common

import torch.nn as nn
import torch


def make_model(args, parent=False):
    return WRANSR(args)


class SelfAttentionBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(SelfAttentionBlock, self).__init__()

        self.conv1 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.conv2 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.conv3 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])

        self.sm = nn.Softmax()

    def forward(self, x):
        in_1 = self.conv1(x)
        in_2 = self.conv2(x)
        in_3 = self.conv3(x)
        # elemen-wise multiple
        out_1 = in_1.mul(in_2)
        out_2 = self.sm(out_1)
        out = out_2.mul(in_3)

        return out


class PlainRAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(PlainRAB, self).__init__()
        childs = [conv(n_feat, n_feat, kernel_size, bias), SelfAttentionBlock(conv, n_feat, kernel_size, bias)]
        self.body = nn.Sequential(childs)

    def forward(self, x):
        x += self.body(x)
        return x


# concatenated ResidualAttentionBlock
class ConcatRAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(ConcatRAB, self).__init__()
        self.convs = conv(n_feat, n_feat, kernel_size, bias)
        self.sqzconv = conv(3 * n_feat, n_feat, kernel_size, bias)
        self.sab = SelfAttentionBlock(conv, n_feat, kernel_size, bias)

    def forward(self, x):
        convlist = list()
        for i in range(3):
            convlist.append(self.convs(x))

        x = self.sqzconv(torch.cat(convlist))
        x += self.sab(x)
        return x


# intermediate ResidualAttentionBlock
class intermediateRAB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(intermediateRAB, self).__init__()
        self.convs = conv(n_feat, n_feat, kernel_size, bias)
        self.sqzconv = conv(2 * n_feat, n_feat, kernel_size, bias)
        self.sab = SelfAttentionBlock(conv, n_feat, kernel_size, bias)

    def forward(self, x):
        x1 = self.convs(x)
        x2 = self.convs(x)
        x3 = self.convs(x)

        y = torch.cat([x1, x2])
        y = self.sab(self.sqzconv(y))
        z = torch.cat([y, x3])
        z = self.sab(self.sqzconv(z))
        z += x
        return z


# embedded ResidualAttentionBlock
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, i_feat, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(ResidualAttentionBlock, self).__init__()
        self.squeeze_convs = conv(i_feat, n_feat, kernel_size, bias)
        self.conv1 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.conv2 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.conv3 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.sab = SelfAttentionBlock(conv, n_feat, kernel_size, bias)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.squeeze_convs(x)
        in_1 = self.conv1(x)
        in_2 = self.conv2(x)
        in_3 = self.conv3(x)
        # elemen-wise multiple
        q = self.sab(in_1)
        k = self.sab(in_2)
        v = self.sab(in_3)

        out_1 = q.mul(k)
        out_2 = self.sm(out_1)
        out = out_2.mul(v)
        out += x
        return out


# dense ResidualAttentionGroup
class ResidualAttentionGroup(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, n_resblocks,
            bias=True, res_scale=1):
        super(ResidualAttentionGroup, self).__init__()
        childblocks = [ResidualAttentionBlock((i + 1) * n_feat, conv, n_feat, kernel_size, bias) for i in
                       range(n_resblocks)]
        self.body = nn.ModuleList(childblocks)

    def forward(self, x):
        reslist = list()
        reslist.append(x)

        for i, r in enumerate(self.body):
            x = torch.cat(reslist, 1)
            x = r(x)
            reslist.append(x)
        return x


class WRANSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(WRANSR, self).__init__()
        n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # head
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # body
        m_body = [ResidualAttentionGroup(conv, n_feats, kernel_size, n_resblocks) for _ in range(n_resgroups)]

        # tail
        m_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)
        ]

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)

        y = res + x

        y = self.tail(y)
        y = self.add_mean(y)

        return y

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


