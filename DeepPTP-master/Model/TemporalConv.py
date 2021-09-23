#导入库
import torch.nn as nn
from torch.nn.utils import weight_norm

#给类实现了修建卷积之后数据的尺寸，使其与输入的尺寸相同
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        #对继承自父类的属性进行初始化
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    #该函数就是第一个数据到倒数第chomp_size的数据
    #其中chomp_size就是padding的值，例如输入的数据是5，padding为1，则会产生6个数据，则只保留前5个数字
    #张量 x 的第一维是批量大小，第二维是通道数量而第三维就是序列长度
    """
    其实这就是一个裁剪的模块，裁剪多出来的padding
    tensor.contiguous()会返回有连续内存的相同张量
    有些tensor并不是占用一整块内存，而是由不同的数据块组成。
    tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()函数，就是把tensor变成在内存中连续分布的形式
    本函数主要是增加padding方式对卷积后的张量做切边而实现因果卷积
    """
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

#该类定义了TCN的基本模块，包含8个部分（两次卷积+修剪数据大小+relu+dropout）
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        """""
        param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """

        # 定义第一轮
        #定义空洞卷积
        #经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        #定义修剪数据
        #裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.chomp1 = Chomp1d(padding)
        #添加激活函数与dropout正则化方法完成第一个卷积
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        #定义第二轮
        #定义空洞卷积
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        #定义修剪数据
        self.chomp2 = Chomp1d(padding)
        #添加激活函数与dropout正则化方法完成第一个卷积
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        #nn.Sequential()函数功能是创建一个有序的容器，将卷积模块的所有组建通过Sequential方法依次堆叠在一起
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        #实现向下采样
        # padding保证了输入序列与输出序列的长度相等，但卷积前的通道数与卷积后的通道数不一定一样。
        # 如果通道数不一样，那么需要对输入x做一个逐元素的一维卷积以使得它的纬度与前面两个卷积相等。
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    #参数初始化
    #初始化为从均值为0，标准差为0.01的正态分布中采样的随机值
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

#TCN主网络
class TemporalConvNet(nn.Module):
    #num_inputs: int， 输入通道数
    #num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
    #kernel_size: int, 卷积核尺寸
    #dropout: float, drop_out比率
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        #计算隐藏层的数目
        num_levels = len(num_channels)
        for i in range(num_levels):
            #空洞卷积的扩张系数若随着网络层级的增加而成指数级增加，则可以增大感受野并不丢弃任何输入序列的元素
            dilation_size = 2 ** i
            #确定每一层的输入通道数（若为第1层，则是就是整个网络的输入通道数；若不是第1层，则为隐藏层的通道数）
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            #确定每一层的输出通道数
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        #*作用是将输入迭代器拆成一个个元素
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

