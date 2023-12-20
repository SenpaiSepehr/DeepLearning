import torch
import torch.nn as nn
import torch.nn.fuctional as F
import torchvision.trnasforms as transforms


## the LBPModule gets called in ChangeFormer.py, in MLPDecoder module


class LBPFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, custom_weights):
        super(LBPFilter, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv.weight = nn.Parameter(custom_weights)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
    

class WeightedSum (nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, custom_weights):
        super(WeightedSum, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv.weight = nn.Parameter(custom_weights)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x        


class LBPModule(nn.Module):
    def __init__(self, input_chan, fixed_weights, learnable_weights):
        super(LBPModule, self).__init__()

        # initiating intensity convs
        self.pointvise_conv = nn.Conv2d(input_chan, out_channels = 1, kernel_size = 1)
        self.intensity_layer = nn.Sequential(
            self.pointvise_conv,
            nn.ReLU()
        )

        # Block #1 - Fixed parameters layers (anchor weights), 8 custom weights in fixed_weights
        self.block1 = nn.ModuleList([
            LBPFilter(1, 8, 3, 3, weights)
            for weights in fixed_weights
        ])

        # Block #2 - Learnable parameter layers, using 1x1 conv to perform weighted sum
        self.block2 = WeightedSum(8, 1, 1, learnable_weights)

    def forward(self, x):

        # Convert to Intensity Tensor
        img_intensity = self.intensity_layer(x)

        # Block #1
        block1_outputs = [custom_filter(img_intensity) for custom_filter in self.block1]
        block1_output = torch.cat(block1_outputs, dim=1) # Expected output (16,8,32,32)

        # Block #2
        block2_output = self.block2(block1_output)  # Output shape: (16, 1, 32, 32)

        # Multiply output of Block#2 with original input
        result = block2_output * x

        return result






# # example
# input = torch.randn(16, 128, 32, 32)
# lbp_module = LBPModule()
# output = lbp_module(input)
# print(output.shape)
