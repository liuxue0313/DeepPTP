# encoding utf-8
import os
import torch
import logging
import argparse

from Train import Train
from Evalution import Evalution

from Model.DeepPTP import DeepPTP

#设置全局变量，控制CPU的使用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#为CPU设置种子用于生成随机数，以使得结果是确定的
#在神经网络中，参数的设置是随机的，但如果每次都是随机则会导致结果的不确定
torch.manual_seed(1111)

#argparse是一个Python模块：命令行选项、参数和子命令解析器。
#程序定义它需要的参数，然后 argparse 将弄清如何从 sys.argv 解析出那些参数。
#argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。
#使用流程:引入模块–>创建解析器–>添加参数–>解析参数

#1、创建解析器：创建一个ArgumentParser对象，其中参数descrip表示在参数帮助文档之后显示的文本
parser = argparse.ArgumentParser(description='DeepPTP')
#2、添加参数：使用add_argumen方法添加参数，其中type是参数类型，default是参数未在命令行中出现时使用的值

#epoch就是整个训练集被训练算法遍历的次数。
parser.add_argument('--epochs', type=int, default=100)
#Batch就是每次送入网络中训练的一部分数据，而Batch Size就是每个batch中训练样本的数量
parser.add_argument('--batchsize', type=int, default=16)
#对于最初输入图片样本的通道数 in_channels 取决于图片的类型，如果是彩色的，即RGB类型，这时候通道数固定为3，如果是灰色的，通道数为1。
#in_channels_S表示空间卷积层的输入通道数（第一次为输入图形通道的大小，后面为上一次卷积out_channels的大小）
parser.add_argument('--in_channels_S', type=int, default=16)
#out_channels_S表示空间卷积层的输出通道数，与上一层的卷积核数量相关
parser.add_argument('--out_channels_S', type=int, default=32)
#kernel_size_S表示卷积核的大小，如果卷积核的长和宽不等，需要用 kernel_h 和 kernel_w 分别设定
parser.add_argument('--kernel_size_S', type=int, default=3)
parser.add_argument('--num_inputs_T', type=int, default=32)
#num_channels_T表示时间卷积层每层的hidden_channel数
parser.add_argument('--num_channels_T', type=list, default=[64] * 5)
#num_outputs_T表示时间卷积层的输入通道数
parser.add_argument('--num_outputs_T', type=int, default=4)
#lr：学习率
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--log_file', type=str, default='./logs/Run.log')
#3、解析参数：把在parser中设置的所有add_argument都返回到args子类中
args = parser.parse_args()

if __name__ == '__main__':
    #设置变量：准确率、精确率、召回率和f1数
    max_accuracy, max_precise, max_recall, max_f1_score = 0., 0., 0., 0.

    #若log_file路径下已经存在文件了，将该文件删除
    if os.path.exists(args.log_file):
        os.remove(args.log_file)

    #logging模块是Python内置的标准模块，主要用于输出运行日志，可以设置输出日志的等级、日志保存路径、日志文件回滚等。
    #参数：filename指定日志文件名称，level指定日志的等级，format指定日志的格式说明
    #%(asctime)s表示打印日志时间，%(name)打印文件名称，%(levelname)s打印日志级别的名称，%(message)s打印日志信息
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #logger：日志对象，logging模块中最基础的对象
    # 用logging.getLogger(name)方法进行初始化，name可以不填。
    log = logging.getLogger(__name__)

    #定义一个DeepPTP网络
    model = DeepPTP(
        in_channels_S=args.in_channels_S,
        out_channels_S=args.out_channels_S,
        kernel_size_S=args.kernel_size_S,
        num_inputs_T=args.num_inputs_T,
        num_channels_T=args.num_channels_T,
        num_outputs_T=args.num_outputs_T
    )
    #判断GPU是否执行该任务（GPU图形处理器）
    if torch.cuda.is_available():
        #在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，而是需要在程序中显示指定。
        #调用model.cuda()，可以将模型加载到GPU上去。
        model.cuda()
    #epoch表示数据集训练的次数
    for epoch in range(args.epochs):
        #通过调用Train来进行模型的训练
        model = Train(model=model, epoch=epoch, batchsize=args.batchsize, lr=args.lr)
        #通过调用Evaluation来进行模型的测试
        max_accuracy, max_precise, max_recall, max_f1_score = \
            Evalution(
                model=model,
                batchsize=args.batchsize,
                max_accuracy=max_accuracy,
                max_precise=max_precise,
                max_recall=max_recall,
                max_f1_score=max_f1_score,
                log=log,
            )
        #清除没用的临时变量
        torch.cuda.empty_cache()




