# encoding utf-8

import torch
import json
from DataLoad import get_loader
import utils
import torch.nn.functional as F


config = json.load(open('./config.json', 'r'))


def Evalution(model, batchsize, max_accuracy, max_precise, max_recall, max_f1_score, log):
    #将模型设置为测试阶段
    model.eval()
    eval_loss = 0
    accuracy = 0
    #定义混淆矩阵（参数是类别的个数）
    confusion_matrix = torch.zeros(4, 4)
    #判断GPU是否执行该任务（GPU是图形处理器）
    if torch.cuda.is_available():
        # 在pytorch中，即使是有GPU的机器，它也不会自动使用GPU，而是需要在程序中显示指定。
        # 调用confusion_matrix.cuda()，可以将该矩阵加载到GPU上去。
        confusion_matrix.cuda()
    #pytorch中不是所有的操作都需要计算图的生成，torch.no_grad可以强制图不生成，生成图方便计算反向传播等
    #with是python的上下文管理器，即当需要进入特定的开始和结束语句时使用
    with torch.no_grad():
        for file in config['eval_set']:
            #读取数据
            dataset = get_loader(file, batchsize)
            # enumerate多用于在for循环中得到计数，利用它可以同时获得索引和值，即需要index和value值的时候可以使用enumerate
            # 输出结果会显示索引和数组，例如list[1,2],输出显示：0,1   1,2
            for idx, parameters in enumerate(dataset):
                # 将获得的数据封装成Variable数据类型（Variable类型是模型输入数据类型）
                # Variable数据类型是对Tensor数据类型的封装，不仅包含Tensor的信息，还有相关的梯度信息，parameter.data就可以取到Tensor数据了
                parameters = utils.to_var(parameters)
                #网络向前传播
                out = model(parameters)
                #reduce参数：默认情况下，设置为True，即根据size_average参数的值决定对每个小批的观察值是进行平均或求和。
                #如果reduce为False，则返回每个批处理元素的损失，不进行平均和求和操作，即忽略size_average参数。
                #size_average默认情况下，设置为True，即对批处理中的每个损失元素进行平均。
                #size_average设置为False，则对每个小批的损失求和。
                #item()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回
                loss = F.nll_loss(out, parameters['direction'], size_average=False).item()
                eval_loss += loss
                #对输出数据的第1维度上求最大值，keepdim=True表示保留原维度
                #取最大值表示预测的分类为该种类型
                pred = out.data.max(1, keepdim=True)[1]
                #torch.squeeze()函数主要对数据的维度进行压缩，去掉维数为1的的维度
                pred = torch.squeeze(pred)
                #构造混淆函数
                #parameters['direction']表示测试数据，pred表示预测值
                #混淆矩阵表示原本为什么值，和预测成了什么值的地方+1
                for i in range(len(pred)):
                    confusion_matrix[parameters['direction'][i]][pred[i]] += 1
                #view_as()返回被视作与给定的tensor相同大小的原tensor，参数是Tensor，view()参数为数值
                #CPU()表示将数据移到CPU上运行
                #根据预测结果和测试结果是否相同计算精确度
                accuracy += pred.eq(parameters['direction'].data.view_as(pred)).cpu().sum()
            #计算精确度、召回率和F1值
            precise, recall, f1_score = utils.CalConfusionMatrix(confusion_matrix)
            #计算正确率
            accuracy_value = accuracy.item() / len(dataset.dataset)
            if accuracy_value > max_accuracy:
                max_accuracy = accuracy_value
            if precise > max_precise:
                max_precise = precise
            if recall > max_recall:
                max_recall = recall
            if f1_score > max_f1_score:
                max_f1_score = f1_score
            #求损失函数的平均值
            eval_loss /= len(dataset.dataset)
            #输出相关的数据
            print('Evalution: Average loss: {:.4f}'.format(eval_loss))
            print(
                'Current Evalution: Accuracy: {}/{} ({:.4f}), Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    accuracy, len(dataset.dataset), accuracy_value, precise, recall, f1_score))
            print(
                'Max Evalution: Accuracy: {:.4f}, Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    max_accuracy, max_precise, max_recall, max_f1_score))

            log.info('Accuracy:{:.4f}, Precise:{:.4f}, Recall:{:.4f}, F1 Score:{:.4f}'.format(accuracy_value, precise, recall, f1_score))

    return max_accuracy, max_precise, max_recall, max_f1_score