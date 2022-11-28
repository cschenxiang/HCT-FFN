import torch
import torch.nn as nn
from torch.nn import functional as F
from model import common
from util.rlutrans import  Mlp, TransBlock
from util.tools import extract_image_patches, reduce_mean, reduce_sum, same_padding, reverse_patches

def make_model(args, parent=False):
    return Rainnet(args)

class OperationLayer(nn.Module):
    def __init__(self, C, stride):
        super(OperationLayer, self).__init__()
        self._ops = nn.ModuleList()
        for o in common.Operations:
            op = common.OPS[o](C, stride, False)
            self._ops.append(op)

        self._out = nn.Sequential(nn.Conv2d(C * len(common.Operations), C, 1, padding=0, bias=False), nn.ReLU())

    def forward(self, x, weights):
        weights = weights.transpose(1, 0)
        states = []
        for w, op in zip(weights, self._ops):
            states.append(op(x) * w.view([-1, 1, 1, 1]))
        return self._out(torch.cat(states[:], dim=1))

class GroupOLs(nn.Module):
    def __init__(self, steps, C):
        super(GroupOLs, self).__init__()
        self.preprocess = common.ReLUConv(C, C, 1, 1, 0, affine=False)
        self._steps = steps
        self._ops = nn.ModuleList()
        self.relu = nn.ReLU()
        stride = 1

        for _ in range(self._steps):
            op = OperationLayer(C, stride)
            self._ops.append(op)

    def forward(self, s0, weights):
        s0 = self.preprocess(s0)
        for i in range(self._steps):
            res = s0
            s0 = self._ops[i](s0, weights[:, i, :])
            s0 = self.relu(s0 + res)
        return s0

class OALayer(nn.Module):
    def __init__(self, channel, k, num_ops):
        super(OALayer, self).__init__()
        self.k = k
        self.num_ops = num_ops
        self.output = k * num_ops
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc = nn.Sequential(
            nn.Linear(channel, self.output * 2),
            nn.ReLU(),
            nn.Linear(self.output * 2, self.k * self.num_ops))

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1)
        y = self.ca_fc(y)
        y = y.view(-1, self.k, self.num_ops)
        return y

def get_residue(tensor , r_dim = 1):
    """
    return residue_channle (RGB)
    """
    # res_channel = []
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel

class convd(nn.Module):
    def __init__(self, inputchannel, outchannel, kernel_size, stride):
        super(convd, self).__init__()
        self.relu = nn.ReLU()
        self.padding = nn.ReflectionPad2d(kernel_size//2)
        self.conv = nn.Conv2d(inputchannel, outchannel, kernel_size, stride)
        self.ins = nn.InstanceNorm2d(outchannel, affine=True)

    def forward(self, x):
        x = self.conv(self.padding(x))
        # x= self.ins(x)
        x = self.relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(Upsample, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
      self.relu = nn.ReLU()

    def forward(self, x, y):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out = self.relu(out)
        out = F.interpolate(out, y.size()[2:])
        return out

class RB(nn.Module):
    def __init__(self, n_feats, nm='in'):
        super(RB, self).__init__()
        module_body = []
        for i in range(2):
            module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
            module_body.append(nn.ReLU())
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()
        self.se = common.SELayer(n_feats, 1)

    def forward(self, x):
        res = self.module_body(x)
        res = self.se(res)
        res += x
        return res

class RIR(nn.Module):
    def __init__(self, n_feats, n_blocks, nm='in'):
        super(RIR, self).__init__()
        module_body = [
            RB(n_feats) for _ in range(n_blocks)
        ]
        module_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.module_body = nn.Sequential(*module_body)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.module_body(x)
        res += x
        return self.relu(res)

class res_ch(nn.Module):
    def __init__(self, n_feats, blocks=2):
        super(res_ch,self).__init__()
        self.conv_init1 = convd(3, n_feats//2, 3, 1)
        self.conv_init2 = convd(n_feats//2, n_feats, 3, 1)
        self.extra = RIR(n_feats, n_blocks=blocks)

    def forward(self,x):
        x = self.conv_init2(self.conv_init1(x))
        x = self.extra(x)
        return x

class Fuse(nn.Module):
    def __init__(self, inchannel=64, outchannel=64):
        super(Fuse, self).__init__()
        self.up = Upsample(inchannel, outchannel, 3, 2)
        self.conv = convd(outchannel, outchannel, 3, 1)
        self.rb = RB(outchannel)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        x = self.up(x, y)
        # x = F.interpolate(x, y.size()[2:])
        # y1 = torch.cat((x, y), dim=1)
        y = x+y
        # y = self.pf(y1) + y

        return self.relu(self.rb(y))

class Prior_Sp(nn.Module):
    def __init__(self, in_dim=32):
        super(Prior_Sp, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        # self.softmax  = nn.Softmax(dim=-1)
        self.sig = nn.Sigmoid()

    def forward(self,x, prior):
        
        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)
        # print(attention.size(),x.size())
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x),dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p),dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out

class DaMoE(nn.Module):
    def __init__(self, n_feats,layer_num ,steps=4):
        super(DaMoE,self).__init__()

        # fuse res
        self.prior = Prior_Sp()
        self.fuse_res = convd(n_feats*2, n_feats, 3, 1)
        self._C = n_feats
        self.num_ops = len(common.Operations)
        self._layer_num = layer_num
        self._steps = steps

        self.layers = nn.ModuleList()
        for _ in range(self._layer_num):
            attention = OALayer(self._C, self._steps, self.num_ops)
            self.layers += [attention]
            layer = GroupOLs(steps, self._C)
            self.layers += [layer]

    def forward(self, x, res_feats):

        x_p, res_feats_p = self.prior(x, res_feats)
        x_s = torch.cat((x_p, res_feats_p),dim=1)
        x1_i = self.fuse_res(x_s)
        for _, layer in enumerate(self.layers):
            if isinstance(layer, OALayer):
                weights = layer(x1_i)
                weights = F.softmax(weights, dim=-1)
            else:
                x1_i = layer(x1_i, weights)

        return x1_i

class BaViT(nn.Module):
    def __init__(self, n_feats, blocks=2):
        super(BaViT, self).__init__()
        # fuse res
        self.prior = Prior_Sp()
        self.fuse_res = convd(n_feats * 2, n_feats, 3, 1)

        self.attention = TransBlock(n_feats, dim=n_feats * 9)
        self.c2 = common.default_conv(n_feats, n_feats, 3)
        # self.attention2 = TransBlock(n_feat=n_feat, dim=n_feat*9)

    def forward(self, x, res_feats):
        x_p, res_feats_p = self.prior(x, res_feats)
        x_s = torch.cat((x_p, res_feats_p), dim=1)
        x1_init = self.fuse_res(x_s)

        y8 = x1_init
        b, c, h, w = y8.shape
        y8 = extract_image_patches(y8, ksizes=[3, 3],
                                   strides=[1, 1],
                                   rates=[1, 1],
                                   padding='same')  # 16*2304*576
        y8 = y8.permute(0, 2, 1)
        out_transf1 = self.attention(y8)
        out_transf1 = self.attention(out_transf1)
        out_transf1 = self.attention(out_transf1)
        out1 = out_transf1.permute(0, 2, 1)
        out1 = reverse_patches(out1, (h, w), (3, 3), 1, 1)
        y9 = self.c2(out1)

        return y9

class Rainnet(nn.Module):
    def __init__(self,args):
        super(Rainnet,self).__init__()
        n_feats = args.n_feats
        blocks = args.n_resblocks
        
        self.conv_init1 = convd(3, n_feats//2, 3, 1)
        self.conv_init2 = convd(n_feats//2, n_feats, 3, 1)
        self.res_extra1 = res_ch(n_feats, blocks)
        self.sub1 = DaMoE(n_feats, 1)
        self.res_extra2 = res_ch(n_feats, blocks)
        self.sub2 = BaViT(n_feats, 1)
        self.res_extra3 = res_ch(n_feats, blocks)
        self.sub3 = DaMoE(n_feats, 1)

        self.ag1 = convd(n_feats*2,n_feats,3,1)
        self.ag2 = convd(n_feats*3,n_feats,3,1)
        self.ag2_en = convd(n_feats*2, n_feats, 3, 1)
        self.ag_en = convd(n_feats*3, n_feats, 3, 1)

        self.output1 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        self.output2 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        self.output3 = nn.Conv2d(n_feats, 3, 3, 1, padding=1)
        
        # self._initialize_weights()

    def forward(self,x):

        res_x = get_residue(x)
        x_init = self.conv_init2(self.conv_init1(x))
        x1 = self.sub1(x_init, self.res_extra1(torch.cat((res_x, res_x, res_x), dim=1))) #+ x   # 1
        out1 = self.output1(x1)
        res_out1 = get_residue(out1)
        x2 = self.sub2(self.ag1(torch.cat((x1,x_init),dim=1)), self.res_extra2(torch.cat((res_out1, res_out1, res_out1), dim=1))) #+ x1 # 2
        x2_ = self.ag2_en(torch.cat([x2,x1], dim=1))
        out2 = self.output2(x2_)
        res_out2 = get_residue(out2)
        x3 = self.sub3(self.ag2(torch.cat((x2,x1,x_init),dim=1)), self.res_extra3(torch.cat((res_out2, res_out2, res_out2), dim=1))) #+ x2 # 3
        x3 = self.ag_en(torch.cat([x3,x2,x1],dim=1))
        out3 = self.output3(x3)

        return out3, out2, out1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


