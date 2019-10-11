import torch
from torch import nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

class Sparse(nn.Module):
    """ #groups independent denses"""
    def __init__(self, in_channels, out_channels, groups = 1, bias = True):
        super().__init__()
        assert(in_channels%groups == 0)
        assert(out_channels%groups == 0)
        in_w = in_channels//groups
        out_w = out_channels//groups
        self.layers = nn.ModuleList([nn.Conv2d(in_w, out_w, kernel_size=1, bias=bias) for i in range(groups)])
        self.groups = groups
        self.reset_parameters()
        
    def forward(self, x, cat = True):
        batch, channels, height, width = x.size()
        x = torch.chunk(x, self.groups, dim = 1)

        out = [self.layers[i](x[i]) for i in range(self.groups)]
        # shape group tensor(b, out_w, h, w)
        
        if cat:
            out = torch.cat(out, dim = 1)
            # (b, out_channels, h, w)
        
        return out

    def reset_parameters(self):
        for i in range(self.groups):
            init.kaiming_normal_(self.layers[i].weight, mode='fan_out', nonlinearity='relu')            
            
class AttentionLite(nn.Module):
    """ Routes information among the densely connected groups and their neighbours via attention"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, heads = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.heads = heads
        in_w = in_channels//groups
        out_w = out_channels//groups
        self.in_w = in_w
        self.out_w = out_w
        
        self.rel_h = nn.Parameter(torch.randn(out_channels //groups// 2, 1, 1, groups, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels //groups// 2, 1, 1, groups, 1, kernel_size), requires_grad=True)

        self.query_conv = Sparse(in_channels, out_channels*heads, groups)
        self.key_conv = Sparse(in_channels, out_channels, groups)
        self.value_conv = Sparse(in_channels, out_channels, groups)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x, cat = False)
        k_out = self.key_conv(padded_x, cat =False) 
        v_out = self.value_conv(padded_x, cat = False)
        # shape tensor(b, out_channels, h, w)
        
        q_out = torch.stack(q_out, dim = -1)
        k_out = torch.stack(k_out, dim = -1)
        v_out = torch.stack(v_out, dim = -1)
        # (b, out_w, h, w, group)
        
        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        # shape (b, out_w, h, w, g, k, k)
        
        k_out_h, k_out_w = k_out.split(self.out_w // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)
        # positional embeddings

        k_out = k_out.contiguous().view(batch, 1, self.out_w, height, width, 1, -1)
        v_out = v_out.contiguous().view(batch, 1, self.out_w, height, width, 1, -1)
        q_out = q_out.view(batch, self.heads, self.out_w, height, width, self.groups, 1)
        # the last dim of k and v is groups*kernel**2
        
        out = q_out * k_out
        # (b, heads, out_w, h, w, g, g*k*k)
        out = out.sum(dim = 2, keepdim = True)
        out = F.softmax(out, dim=-1)
        # out :(b, heads, 1, h, w, g, g*k*k)
        # v_out: (b, 1, out_w. h, w, 1, g*k*k)
        out = out*v_out
        # (b, heads, out_w, h, w, g, g*k*k)
        out = out.sum(dim = 1)
        out = out.sum(dim = -1)
        # (b, out_w, h, w, g)
        out = out.transpose(1, -1)
        out = out.reshape(batch, -1, height, width)
        return out

    def reset_parameters(self):
        self.query_conv.reset_parameters()
        self.value_conv.reset_parameters()
        self.key_conv.reset_parameters()

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
        
def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

class Model(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, groups = 1,
                 kernel_size = 1, stride = 1, padding = 0, heads = 1):
        super().__init__()
        self.sparse1 = Sparse(in_channels, inter_channels, groups)
        self.BN = nn.BatchNorm2d(inter_channels)
        self.attention = AttentionLite(inter_channels, inter_channels, kernel_size, stride, padding, groups, heads)
        self.sparse2 = Sparse(inter_channels, out_channels, groups)
        self.reset_parameters()
        
    def forward(self, x):
        out = self.sparse1(x)
        out = self.BN(out)
        out = F.relu(out)
        out = self.attention(out)
        out = self.BN(out)
        out = F.relu(out)
        out = self.sparse2(out)
        
        return x+out

    def reset_parameters(self):
        self.sparse1.reset_parameters()
        self.attention.reset_parameters()
        self.sparse2.reset_parameters()