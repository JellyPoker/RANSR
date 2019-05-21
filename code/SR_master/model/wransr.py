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

        self.conv1 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True)])
        self.conv2 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True)])
        self.conv3 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True)])

        self.sm = nn.Softmax()

    def forward(self, x):
        in_1 = self.conv1(x)
        in_2 = self.conv2(x)
        in_3 = self.conv3(x)
        # elemen-wise multiple

        out_1 = in_1.mul(in_2)
        out_2 = self.sm(out_1)
        out = out_2.mul(in_3)
        out = out + x
        return out

class EmbeddedAttentionBlock(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(EmbeddedAttentionBlock, self).__init__()

        self.conv1 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.Softmax()])
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
        out = out + x
        return out



class plainRAB(nn.Module):
    def __init__(
            self,i_feat, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(plainRAB, self).__init__()
        self.squeeze_convs = nn.Sequential(*[conv(i_feat, n_feat, kernel_size, bias)])
        self.conv1 = nn.Sequential(*[conv(n_feat,n_feat,kernel_size,bias),nn.Softmax()])
        self.conv2 = nn.Sequential(*[conv(n_feat,n_feat,kernel_size,bias)])

    def forward(self, x):
        x = self.squeeze_convs(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y = x1.mul(x2)
       # y += x2
        y += x
        return y


# concatenated ResidualAttentionBlock
class concatRAB(nn.Module):
    def __init__(
            self,i_feat, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(concatRAB, self).__init__()
        self.squeeze_convs = conv(i_feat, n_feat, kernel_size, bias)

        self.conv1 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.conv2 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.conv3 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.sqzconv = nn.Sequential(*[conv(2 * n_feat, n_feat, kernel_size, bias),nn.Softmax()])


    def forward(self, x):
        x = self.squeeze_convs(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        y = torch.cat([x1,x2],1)
        y = self.sqzconv(y)
        y = x3.mul(y)
        y += x3
        y += x
        return y
    
class embeddedRAB(nn.Module):
    def __init__(
            self, i_feat, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(embeddedRAB, self).__init__()
        self.squeeze_convs = nn.Sequential(*[conv(i_feat, n_feat, kernel_size, bias)])
        self.conv1 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.Softmax()])
        self.conv2 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.conv3 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias)])
        self.sm = nn.Softmax()


    def forward(self, x):
        x = self.squeeze_convs(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        y = x1.mul(x2)
        y += x2
        y = self.sm(y)
        y = x3.mul(y)
        y += x3
        y += x
        return y


# intermediate ResidualAttentionBlock
class intermediateRAB(nn.Module):
    def __init__(
            self, i_feat, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(intermediateRAB, self).__init__()
        self.squeeze_convs = conv(i_feat, n_feat, kernel_size, bias)
        self.conv1 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True)])
        self.conv2 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True)])
        self.conv3 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True)])
        self.sqzconv1 = conv(3 * n_feat, n_feat, kernel_size, bias)
        #self.sqzconv2 = conv(2 * n_feat, n_feat, kernel_size, bias)
        self.eab1 = EmbeddedAttentionBlock(conv, 2*n_feat, kernel_size, bias)
        self.eab2 = EmbeddedAttentionBlock(conv, 3*n_feat, kernel_size, bias)

    def forward(self, x):
        x = self.squeeze_convs(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        y = torch.cat([x1, x2],1)
        y = self.eab1(y)

        z = torch.cat([y, x3],1)
        z = self.eab2(z)

        z = self.sqzconv1(z)
        z += x
        return z


# embedded ResidualAttentionBlock
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self, i_feat, conv, n_feat, kernel_size,
            bias=True, res_scale=1):
        super(ResidualAttentionBlock, self).__init__()
        self.squeeze_convs = conv(i_feat, n_feat, kernel_size, bias)

        self.branch1 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True),
                                     EmbeddedAttentionBlock(conv, n_feat, kernel_size, bias),nn.Softmax()])
        self.branch2 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True),
                                     EmbeddedAttentionBlock(conv, n_feat, kernel_size, bias)])
        self.branch3 = nn.Sequential(*[conv(n_feat, n_feat, kernel_size, bias),nn.ReLU(inplace=True),
                                     EmbeddedAttentionBlock(conv, n_feat, kernel_size, bias)])
        self.sm = nn.Softmax()

    def forward(self, x):
        x = self.squeeze_convs(x)
        q = self.branch1(x)
        k = self.branch2(x)
        v = self.branch3(x)
        # elemen-wise multiple

        out_1 = q.mul(k)
        out_2 = self.sm(out_1)
        out = out_2.mul(v)
        out += x
        return out


# dense ResidualAttentionGroup
class ResidualAttentionGroup(nn.Module):
    def __init__(
            self, rabtype, conv, n_feat, kernel_size, n_resblocks,
            bias=True, res_scale=1):
        super(ResidualAttentionGroup, self).__init__()
        if rabtype == 'c':
            block = embeddedRAB
        elif rabtype == 'b':
            block = concatRAB
        elif rabtype == 'a':
            block = plainRAB
        else:
            print('wrong parameter for rab type')

        childblocks = [block((i + 1) * n_feat, conv, n_feat, kernel_size, bias) for i in
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
        rabtype=args.rabtype

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        # head
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # body
        m_body = [ResidualAttentionGroup(rabtype,conv, n_feats, kernel_size, n_resblocks) for _ in range(n_resgroups)]

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


