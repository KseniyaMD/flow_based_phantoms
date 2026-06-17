import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp

# Different affine nets for input in the first and second coupling blocks

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
        x = torch.cat(x, dim=1)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def reverse(self, x):
        return x[-1]


class WeightedConcatLayer(nn.Module):
    def __init__(self, channel, activate='softmax'):
        super(WeightedConcatLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(channel), requires_grad=True)
        if activate == 'softmax':
            self.activate = nn.Softmax(-1)
        elif activate == 'sigmoid':
            self.activate = nn.Sigmoid()

    def forward(self, x):
        weights = self.activate(self.weight)
        out = []
        for idx, feat in enumerate(x):
            out.append(weights[idx] * feat)
        return torch.cat(out, dim=1)

    def reverse(self, x):
        weights = self.activate(self.weight)
        out = 0
        for idx, feat in enumerate(x):
            out += weights[idx] * feat
        return out


class FlatConcatLayer(nn.Module):
    def __init__(self):
        super(FlatConcatLayer, self).__init__()

    def forward(self, x):
        return torch.cat(x, dim=1)

    def reverse(self, x):
        return x[-1]


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def calc_mean_std(self, feat, eps=1e-5):
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, content, style_mean, style_std):
        assert style_mean is not None
        assert style_std is not None

        size = content.size()
        content_mean, content_std = self.calc_mean_std(content)
        style_mean = style_mean.reshape(size[0], content_mean.shape[1], 1, 1)
        style_std = style_std.reshape(size[0], content_mean.shape[1], 1, 1)
        normalized_feat = (content - content_mean.expand(size)) / content_std.expand(size)
        sum_mean = style_mean.expand(size)
        sum_std = style_std.expand(size)
        return normalized_feat * sum_std + sum_mean


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            # self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if norm == 'sn':
            self.conv = SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):
        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LB(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LB, self).__init__()
        use_bias = True
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=use_bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim, n_blk, norm='none', activ='relu'):
        super(MLP, self).__init__()
        self.model = []
        self.model += [LB(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LB(dim, dim, norm=norm, activation=activ)]
        self.model += [LB(dim, output_dim, norm='none', activation='none')]  # no output activations
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.model(x)
        return x


class StyleEncoder(nn.Module):
    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):
        super(StyleEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]  # global average pooling
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ChannelAffineNet(nn.Module):
    def __init__(self, in_ch=1, base_dim=16, out_dim=10, seq_dim=3):
        super().__init__()

        # 1) Простая MLP для превращения TE/TR/ESP → в базовый spatial эмбеддинг
        self.embed1 = nn.Sequential(
            nn.Linear(seq_dim, base_dim * 4),
            nn.ReLU(),
            nn.Linear(base_dim * 4, base_dim)
        )

        # 2) 1-й сверточный блок
        self.conv1 = nn.Conv2d(in_ch + base_dim, base_dim, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(base_dim)

        # 3) Второй embedding
        self.embed2 = nn.Sequential(
            nn.Linear(seq_dim, base_dim * 4),
            nn.ReLU(),
            nn.Linear(base_dim * 4, base_dim)
        )

        # 4) Второй сверточный блок
        self.conv2 = nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(base_dim * 2)

        # 5) Финальный выходной слой
        self.conv_out = nn.Conv2d(base_dim * 2 + base_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, x, seq_param):
        """
        x: [B, 1, H, W]     — один канал MRI
        seq_param: [B, 3]    — TE, TR, ESP
        """

        B, C, H, W = x.shape
        #assert C == 1

        # ————————————————
        # Embed params → spatial
        # ————————————————

        # 1) initial param embedding
        e1 = self.embed1(seq_param)  # [B, base_dim]
        e1 = e1.unsqueeze(-1).unsqueeze(-1)  # [B, base_dim, 1, 1]
        e1 = e1.expand(B, e1.size(1), H, W)  # [B, base_dim, H, W]

        # 2) concat to input channel
        out = torch.cat([x, e1], dim=1)  # [B, 1+base_dim, H, W]
        out = self.conv1(out)
        out = self.norm1(out)
        out = F.relu(out)

        # ————————————————
        # second param embedding
        # ————————————————

        e2 = self.embed2(seq_param)  # [B, base_dim]
        e2 = e2.unsqueeze(-1).unsqueeze(-1)  # [B, base_dim, 1, 1]
        e2 = e2.expand(B, e2.size(1), H, W)  # [B, base_dim, H, W]

        # concat to intermediate features
        out = torch.cat([out, e2], dim=1)  # [B, base_dim*2, H, W]
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)

        # ————————————————
        # final output
        # ————————————————

        final = torch.cat([out, e2], dim=1)  # add condition again if helpful
        final = self.conv_out(final)  # [B, out_dim, H, W]

        return final

class MultiChannelAffine(nn.Module):
    def __init__(self, in_channels=3, seq_dim=3, out_per=10, base_dim=16, part_channels=1):
        super().__init__()
        self.n = in_channels
        self.part_channels = part_channels
        self.nets = nn.ModuleList([
            ChannelAffineNet(in_ch=part_channels, base_dim=base_dim, out_dim=out_per, seq_dim=seq_dim)
            for _ in range(in_channels//part_channels)
        ])

    def forward(self, x, seq_params):
        """
        x:          [B, 3, H, W]
        seq_params: [B, 3, 3]  — для каждого MRI-канала свои TE,TR,ESP
        """

        outs = []
        for i, net in enumerate(self.nets):
            xi = x[:, i:i+self.part_channels, :, :]              # [B,1,H,W]
            pi = seq_params[:, i, :]            # [B,3]
            yi = net(xi, pi)                    # [B,out_per,H,W]
            outs.append(yi)

        # concat по канальному измерению
        out = torch.cat(outs, dim=1)            # [B, 3*out_per, H, W]
        return out

class HierarchyCoupling(nn.Module):
    def __init__(self, in_channel, out_channel, weight_type='fixed', seq_dim=3, out_per=10, base_dim=16, part_channels=1):
        super(HierarchyCoupling, self).__init__()
        self.feat = None
        self.out_channel = out_channel
        self.in_channel = in_channel


        self.affine_net = MultiChannelAffine(in_channels=in_channel, seq_dim=seq_dim, out_per=out_per, base_dim=base_dim, part_channels=part_channels)
        self.adain = AdaIN()
        self.style_mlp = MLP(8, out_channel * 2, out_channel * 3, 3, norm='none', activ='relu')

        self.splits = self.out_channel // self.in_channel
        self.weight_type = weight_type
        self.fixed_weight = 0.5
        if self.weight_type == 'softmax' or self.weight_type == 'sigmoid':
            self.weight = WeightedConcatLayer(channel=self.splits, activate=self.weight_type)
        elif self.weight_type == 'attention':
            self.weight = SELayer(channel=self.out_channel, reduction=self.splits)
        elif self.weight_type == 'fixed':
            self.weight = FlatConcatLayer()
        elif self.weight_type == 'learned':
            self.weight = FlatConcatLayer()
            self.fixed_weight = nn.Parameter(torch.tensor(self.fixed_weight))
            self.fixed_weight.requires_grad = True
        else:
            raise NotImplementedError('Error: type {} for weight is not implemented.'.format(self.weight_type))

    def forward(self, input, seq_params):
        b, c, h, w = input.shape
        n_channel = c

        feature = self.affine_net(input, seq_params)


        self.feat = feature

        output_list = []
        out = input - feature[:, 0:n_channel]
        output_list.append(out)
        tmp_out = out
        for i in range(1, self.splits):
            tmp_out = tmp_out - feature[:, i * n_channel:(i + 1) * n_channel]
            output_list.append(tmp_out)

        return self.weight(output_list)

    def reverse(self, input, style):
        feature = self.feat

        pred_style = self.style_mlp(style)
        mean, std = pred_style.chunk(2, 1)
        input = self.adain(input, mean, std)

        output_list = []
        tmp_out = input[:, -self.in_channel:] + feature[:, -self.in_channel:]
        output_list.append(tmp_out)
        for i in range(self.splits - 1, 0, -1):
            tmp_out = self.fixed_weight * (tmp_out + input[:, (i - 1) * self.in_channel:i * self.in_channel]) + feature[
                                                                                                                :, (
                                                                                                                               i - 1) * self.in_channel:i * self.in_channel] * (
                                  1 - self.fixed_weight)
            output_list.append(tmp_out)

        return self.weight.reverse(output_list)


class HierarchyCoupling_base(nn.Module):
    def __init__(self, in_channel, out_channel, weight_type='fixed'):
        super(HierarchyCoupling_base, self).__init__()
        self.feat = None
        self.out_channel = out_channel
        self.in_channel = in_channel



        self.affine_net = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel * 2, kernel_size=3, stride=1, padding=1,
                      dilation=1),
            nn.InstanceNorm2d(in_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel * 2, out_channels=in_channel * 2, kernel_size=3, stride=1, padding=1,
                      dilation=1),
            nn.InstanceNorm2d(in_channel * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channel * 2, out_channels=out_channel, kernel_size=3, stride=1, padding=1,
                      dilation=1),
            nn.ReLU(inplace=True),
        )
        self.adain = AdaIN()
        self.style_mlp = MLP(8, out_channel * 2, out_channel * 3, 3, norm='none', activ='relu')

        self.splits = self.out_channel // self.in_channel
        self.weight_type = weight_type
        self.fixed_weight = 0.5
        if self.weight_type == 'softmax' or self.weight_type == 'sigmoid':
            self.weight = WeightedConcatLayer(channel=self.splits, activate=self.weight_type)
        elif self.weight_type == 'attention':
            self.weight = SELayer(channel=self.out_channel, reduction=self.splits)
        elif self.weight_type == 'fixed':
            self.weight = FlatConcatLayer()
        elif self.weight_type == 'learned':
            self.weight = FlatConcatLayer()
            self.fixed_weight = nn.Parameter(torch.tensor(self.fixed_weight))
            self.fixed_weight.requires_grad = True
        else:
            raise NotImplementedError('Error: type {} for weight is not implemented.'.format(self.weight_type))

    def forward(self, input, seq_params):
        b_size, n_channel, height, width = input.shape

        feature = self.affine_net(input)
        self.feat = feature

        output_list = []
        out = input - feature[:, 0:n_channel]
        output_list.append(out)
        tmp_out = out
        for i in range(1, self.splits):
            tmp_out = tmp_out - feature[:, i * n_channel:(i + 1) * n_channel]
            output_list.append(tmp_out)

        return self.weight(output_list)

    def reverse(self, input, style):
        feature = self.feat

        pred_style = self.style_mlp(style)
        mean, std = pred_style.chunk(2, 1)
        input = self.adain(input, mean, std)

        output_list = []
        tmp_out = input[:, -self.in_channel:] + feature[:, -self.in_channel:]
        output_list.append(tmp_out)
        for i in range(self.splits - 1, 0, -1):
            tmp_out = self.fixed_weight * (tmp_out + input[:, (i - 1) * self.in_channel:i * self.in_channel]) + feature[
                                                                                                                :, (
                                                                                                                               i - 1) * self.in_channel:i * self.in_channel] * (
                                  1 - self.fixed_weight)
            output_list.append(tmp_out)

        return self.weight.reverse(output_list)


class HierarchyFlow(nn.Module):
    def __init__(self, pad_size=10, in_channel=3, out_channels=[30, 120], weight_type='fixed'):
        super(HierarchyFlow, self).__init__()
        self.pad_size = pad_size
        self.num_block = len(out_channels)
        self.in_channels = [in_channel]
        self.out_channels = out_channels

        self.padding = torch.nn.ReflectionPad2d(self.pad_size)
        self.blocks = nn.ModuleList()
        # for i in range(self.num_block):
        # self.blocks.append(HierarchyCoupling(in_channel=self.in_channels[i], out_channel=self.out_channels[i],
        # weight_type=weight_type))
        # self.in_channels.append(self.out_channels[i])

        self.blocks.append(HierarchyCoupling(in_channel=self.in_channels[0], out_channel=self.out_channels[0],
                                             weight_type=weight_type))
        self.in_channels.append(self.out_channels[0])
        self.blocks.append(HierarchyCoupling(in_channel=self.in_channels[1], out_channel=self.out_channels[1],
                                                  weight_type=weight_type, out_per=self.out_channels[1]//3, part_channels=self.in_channels[1]//3))
        self.style_net = StyleEncoder(n_downsample=2, input_dim=3, dim=64, style_dim=8, norm='none', activ='relu',
                                      pad_type='reflect')

    def forward(self, content, style, in_seq_params):
        style_feat = self.style_net(style)
        content = self.padding(content)
        b_size, n_channel, height, width = content.shape
        for i in range(self.num_block):
            content = self.blocks[i](content, in_seq_params)
        for i in range(self.num_block - 1, -1, -1):
            content = self.blocks[i].reverse(content, style_feat)
        content = content[:, :, self.pad_size:height - self.pad_size, self.pad_size:width - self.pad_size]
        return content