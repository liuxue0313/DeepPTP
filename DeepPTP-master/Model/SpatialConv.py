# encoding utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F

#创建空间卷积层
class SpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SpatialConv, self).__init__()
        #定义一维卷积层（一维卷积一般用来处理文本）
        #in_channels(int) – 输入信号的通道。有多少个in_channels，就需要多少个卷积核通道，即1个卷积核中有几维
        #out_channels(int) – 卷积产生的通道。有多少个out_channels，就需要多少个卷积核
        #kernel_size(int or tuple) - 卷积核的尺寸。
        #卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kernel_size*in_channels
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        #nn.Linear()函数用来设置全连接层，参数分别为输入变量和输出变量的长度
        #全连接层：每一层的每个结点逗鱼上一层的所有结点相连
        self.linear = nn.Linear(2, 16)

    def forward(self, parameters):
        #torch.unsqueeze是对数据进行压缩
        X, Y = torch.unsqueeze(parameters['X'], dim=2), torch.unsqueeze(parameters['Y'], dim=2)
        #torch.cat()在给定维度上对输入的张量序列seq 进行连接操作
        #参数inputs：带连接的张量序列（任何tensor类型的序列）；参数dim：选择的扩维，沿着此维度进行链接
        locations = torch.cat((X, Y), 2)
        #F.tanh()表示激活函数。将进行全连接后的数据使用tanh函数激活
        #permute()函数可以交换tensor的维度
        locations_linear = F.tanh(self.linear(locations)).permute(0, 2, 1)
        #将全连接输出的数据输出到卷积神经网络中，在使用elu函数进行激活
        out = F.elu(self.conv(locations_linear))
        return out