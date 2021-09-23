# encoding utf-8
import json
import torch.nn.functional as F
#torch.optim是一个实现了多种优化算法的包
import torch.optim as optim
import utils
from DataLoad import get_loader

config = json.load(open('./config.json', 'r'))

#训练数据：参数分别为model为DeepPTP，epoch就是整个训练集被训练算法遍历的次数
#batchsize表示每次执行数据的大小，lr表示学习率
def Train(model, epoch, batchsize, lr):
    #将模型设置为训练模式（有些layer在训练和测试阶段不同）
    model.train()
    #torch.optim是一个实现了多种优化算法的包
    #使用优化算法，需要先创建一个优化器对象optimizer，用来保存当前的状态，并能够根据计算得到的梯度来更新参数。
    #要构建一个优化器optimizer，首先定义一个可进行迭代优化的包含了所有参数（所有的参数必须是变量s）的列表。
    #然后，指定程序优化特定的选项，例如学习速率，权重衰减等。
    #model.parameters参数：可用于迭代优化的参数或者定义参数组的dicts。
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #train loss是训练数据上的损失，衡量模型在训练集上的拟合能力
    train_loss = 0
    for file in config['train_set']:
        #调用DataLoad中的get_loader函数，完成数据采样，返回一个batch，保存到dataset中
        dataset = get_loader(file, batchsize)
        #enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值，即需要index和value值的时候可以使用enumerate
        #输出结果会显示索引和数组，例如list[1,2],输出显示：0,1   1,2
        for idx, parameters in enumerate(dataset):
            #将获得的数据封装成Variable数据类型
            #Variable数据类型是对Tensor数据类型的封装，不仅包含Tensor的信息，还有相关的梯度信息，parameter.data就可以取到Tensor数据了
            parameters = utils.to_var(parameters)
            #网络向前传播
            out = model(parameters)
            #定义损失函数：logsoftmax，参数分别为目标数据和输出数据
            loss = F.nll_loss(out, parameters['direction'])
            #将网络中所有的梯度设置为0
            optimizer.zero_grad()
            #回传失误
            loss.backward()
            #回传损失过程中会计算梯度，然后需要根据这些梯度更新参数
            #optimizer.step()就是用来更新参数的。
            optimizer.step()

            train_loss += loss
            if idx > 0 and idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, idx * batchsize, len(dataset.dataset),
                        100. * idx / len(dataset), train_loss.item() / 10))
                train_loss = 0
    return model