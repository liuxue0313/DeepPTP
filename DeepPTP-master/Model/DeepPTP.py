# encoding utf-8
import torch
import torch.nn as nn
#同一个包下引用其他的类，直接用.+程序名即可
from .TemporalConv import TemporalConvNet
from .SpatialConv import SpatialConv
import torch.nn.functional as F


class DeepPTP(nn.Module):
    def __init__(self, in_channels_S, out_channels_S, kernel_size_S, num_inputs_T, num_channels_T, num_outputs_T):
        super(DeepPTP, self).__init__()
        #定义一个空间卷积层
        self.SpatialConv = SpatialConv(in_channels=in_channels_S, out_channels=out_channels_S, kernel_size=kernel_size_S)
        #定义一个时间卷积层
        self.TemporalConv = TemporalConvNet(num_inputs=num_inputs_T + 2, num_channels=num_channels_T)
        #定义隐藏层（一维卷积层）
        self.implicit_net = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size_S)
        #定义全连接层
        self.full_connect = nn.Linear(num_channels_T[-1], num_outputs_T)

    def forward(self, parameters):
        #调用空间卷积网络
        out = self.SpatialConv(parameters)

        #将辅助因素与行人之间的关系嵌入到数据中，辅助因素考虑距离和时间
        distance = torch.unsqueeze(parameters['distance'], dim=2).permute(0, 2, 1)
        distance = self.implicit_net(distance).permute(0, 2, 1)
        out = torch.cat((out.permute(0, 2, 1), distance), dim=2)

        times = torch.unsqueeze(parameters['time_gap'], dim=2).permute(0, 2, 1)
        times = self.implicit_net(times).permute(0, 2, 1)
        out = torch.cat((out, times), dim=2).permute(0, 2, 1)

        #调用时间卷积网络
        out = self.TemporalConv(out)
        #调用全连接网络
        out = self.full_connect(out[:, :, -1])

        #返回分类后的结果（使用log_softmax进行分类）
        return F.log_softmax(out, dim=1)
