import torch
import torch.nn as nn
import sys
import numpy as np
# import torchvision
# from torchvision.models.resnet import Bottleneck, BasicBlock, conv1x1

class MyNet(nn.Module):
    def __init__(self,symbol_name):
        super(MyNet, self).__init__()
        self.conv1 = nn.Sequential(  # 3x501x501 每卷一次像素1/2，第一个是通道数，BCHW，C-通道-越来越大，HW-长宽-越来越小
            nn.Conv2d(
                in_channels=3,  # input image channels
                out_channels=16,
                kernel_size=3,  # 卷积核大小，肯定不用改
                stride=2,  # 卷积步长很重要，与输出尺寸有关
                padding=1,  # 不用动
            ),  # 16x256x256
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 16x128x128
        )
        self.conv2 = nn.Sequential(  #
            nn.Conv2d(16, 32, 3, 2, 1),  # 32x64x64
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32x32
        )
        self.conv3 = nn.Sequential(  #
            nn.Conv2d(32, 32, 3, 2, 1),  # 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x8x8
        )
        self.conv4 = nn.Sequential(  #
            nn.Conv2d(32, 32, 3, 2, 1),  # 32x4x4
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x2x2
        )

        self.out = nn.Linear(32 * 2 * 2, 4)  # 紧接CNN的图像信息，连接输出
        self.input = nn.Sequential(nn.Linear(7, 32), nn.Linear(32, 4))  # 数字的输出通道


    def forward(self, x,input):
        x=x/255
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        # print(x1.shape)
        x1 = x1.view(x1.size(0), -1)  # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)\
        # print(x1.shape)
        output1 = self.out(x1)   # 图像卷积之后得结果
        output2 = self.input(input)
        print(output2.shape)
        if output2.shape == (4,):
            output2.unsqueeze_(0)
            print(output2.shape)
        output=torch.cat((output1, output2), 1)
        #device = torch.device('cuda:0')
        #a=torch.tensor([0,0,0,0,1,1,1,1], device=device)
        #b=torch.tensor([1,1,1,1,0,0,0,0], device=device)
        #output = torch.mul(a,output1) + torch.mul(b,output2)  # 网络结构设计
        return output
def get_symbol(symbol_name='MyNet'):
    return MyNet(symbol_name)

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

if __name__ == '__main__':
    MyNet = MyNet(symbol_name='MyNet')
    MyNet.eval()

    x = 255*torch.randn((1, 3, 501, 501))
    input_x = torch.randn((1, 7))
    label_x = torch.randn((1, 8))
    x_Out = MyNet(x,input_x)
    print(f'input size: {x.shape}')

    criterion_l2 = nn.MSELoss()
    loss2 = criterion_l2(x_Out, label_x)
    print(loss2)
    print(x_Out.shape)

    sys.exit()