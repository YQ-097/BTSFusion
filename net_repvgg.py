import numpy as np
import torch
import torch.nn as nn
from args_fusion import args
import CBAM
from repvgg import RepVGGBlock,repvgg_model_convert


class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.is_last is False:
            out = self.relu(out)
        return out


# Shuffle network
class Repvgg_net(nn.Module):
    def __init__(self, input_nc=args.input_nc, output_nc=args.output_nc, deploy=False, use_se=False):
        super(Repvgg_net, self).__init__()
        regvppblock = RepVGGBlock
        nb_filter = [16, 32, 64, 32, 16]
        kernel_size = 3
        stride = 1
        self.deploy = deploy
        self.use_se = use_se

        self.relu = nn.ReLU(inplace=True)

        #vi encode
        self.conv_vi = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.RepVgg_vi1 = RepVGGBlock(in_channels=nb_filter[0], out_channels=nb_filter[1], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.RepVgg_vi2 = RepVGGBlock(in_channels=nb_filter[1], out_channels=nb_filter[1], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)

        #ir encode
        self.conv_ir = ConvLayer(input_nc, nb_filter[0], kernel_size, stride)
        self.RepVgg_ir1 = RepVGGBlock(in_channels=nb_filter[0], out_channels=nb_filter[1], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.RepVgg_ir2 = RepVGGBlock(in_channels=nb_filter[1], out_channels=nb_filter[1], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)


        self.cbam = CBAM.CBAMBlock(nb_filter[1]*2, reduction=4, kernel_size=kernel_size) #out 128 yuan


        #decode
        self.RepVgg_decode1 = RepVGGBlock(in_channels=nb_filter[1]*2, out_channels=nb_filter[1], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.RepVgg_decode3 = RepVGGBlock(in_channels=nb_filter[3], out_channels=nb_filter[4], kernel_size=3, stride=1, padding=1, deploy=self.deploy, use_se=self.use_se)
        self.conv_out = ConvLayer(nb_filter[4], output_nc, kernel_size, stride)
        
    def encoder(self, input):
        x1 = self.conv_vi(input)
        x2 = self.RepVgg_vi1(x1)
        x3 = self.RepVgg_vi2(x2)

        return [x3]

    def encoder_ir(self, input):
        x1 = self.conv_ir(input)
        x2 = self.RepVgg_ir1(x1)
        x3 = self.RepVgg_ir2(x2)
        return [x3]

    def fusion(self, en1, en2):
        x = torch.cat([en1[0], en2[0]], 1)
        x = self.cbam(x)
        return [x]

    def decoder(self, f_en):
        x1 = self.RepVgg_decode1(f_en[0])
        x3 = self.RepVgg_decode3(x1)
        output = self.conv_out(x3)
        return [output]


if __name__ == '__main__':
    densefuse_model = Repvgg_net(args.input_nc, args.output_nc, deploy=False)
    densefuse_model.load_state_dict(torch.load("./models/BTSFusion.model"))
    repvgg_model_convert(densefuse_model, save_path="./models/test/test1_model.model", do_copy=True)