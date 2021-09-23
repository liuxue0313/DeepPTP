# encoding utf-8

from torch.utils.data import Dataset, DataLoader
import ujson
import numpy as np
import torch
import utils
import pandas as pd
import csv

class MySet(Dataset):
    # 该方法主要完成三个功能：
    #1、从文件中读取数据
    #2、将数据转为dict型保存
    #3、求的每个dict数据的长度
    def __init__(self, input_file):
        #从文件中读取数据
        self.content = open('./data/' + input_file, 'r').readlines()
        #使用循环，将str的数据转换为dict格式（使用函数ujson.loads），再保存到self.content中
        #map()函数，接收一个list和f，将f作用于每一个list的元素上。
        #lambda表示匿名函数
        self.content = list(map(lambda x: ujson.loads(x), self.content))
        # 保存了content中每个dict元素中‘X’对应value的长度
        self.lengths = list(map(lambda x: len(x['X']), self.content))

    #获取指定的元素
    def __getitem__(self, idx):
        return self.content[idx]

    #获取dict的大小
    def __len__(self):
        return len(self.content)



#将一个list的sample组成一个mini-batch的函数
#collate_fn函数将数据整理成Batch_data形式
def collate_fn(data):
    keys = ['time_gap', 'X', 'Y', 'direction', 'distance', 'dist']

    parameters = {}
    #np.asarray函数将输入的数据转化为矩阵的形式
    lens = np.asarray([len(item['X']) for item in data])

    for key in keys:
        if key in ['time_gap', 'X', 'Y', 'distance']:
            seqs = np.asarray([item[key] for item in data])
            #np.arange函数，生成array对象
            #数组切片取None，会多出一个维度
            mask = np.arange(lens.max()) < lens[:, None]
            #构造一个0矩阵，大小为mask的大小
            padded = np.zeros(mask.shape, dtype=np.float32)
            #concatenate()函数根据指定的维度，对一个元组、列表中的list或者ndarray进行连接
            padded[mask] = np.concatenate(seqs)
            padded = utils.Z_Score(padded)

            padded = torch.from_numpy(padded).float()
            parameters[key] = padded

        elif key == 'direction':
            parameters[key] = torch.from_numpy(np.asarray([item[key] for item in data])).type(torch.long)

        elif key == 'dist':
            x = np.asarray([item[key] for item in data])
            x = utils.Z_Score(x)
            parameters[key] = torch.from_numpy(x).type(torch.long)

    lens = lens.tolist()
    parameters['lens'] = lens

    return parameters

#该类实现数据的采样功能
#Sampler类主要觉得读那些数据
class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = list(range(self.count))

    def __iter__(self):
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

#该方法实现数据采样功能
#分为四步：1、调用该方法
#2、创建数据集
#3、创建采样策略类的对象
#、调用函数进行数据采样，返回一个batch
def get_loader(file, batch_size):
    #dataset返回的为一个list类，但是list不能作为模型的输入
    #torch.utils.data.DataLoader类可以将list类型的输入数据封装成Tensor数据格式，以备模型使用
    #DataSet决定从哪读取数据
    dataset = MySet(input_file=file)
    #自定义从数据集中采样的策略
    #sampler决定读那些数据
    batch_sampler = BatchSampler(dataset, batch_size)
    #DataLoser类参数：dataset传入数据集、batch_size每个batch有几个样本
    #collate_fn将一个list的sample组成一个mini-batch的函数
    #num_workers表示有几个进程来处理data loading，默认为0，0表示所有的数据都会被load进主进程
    #batch_samper表示与sampler类似（自定义从数据集中取样本的策略）
    #但是batch_sample返回batch的indices（索引），迭代器
    #一旦指定了这个参数，那么batch_size,shuffle（每个epoch开始对数据进行重新排序）,sampler,drop_last就不能再制定了
    #pin_memory=True，那么data loader将会在返回batch之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
    data_loader = DataLoader(dataset=dataset, batch_size=1, collate_fn=lambda x: collate_fn(x), num_workers=0,
                             batch_sampler=batch_sampler, pin_memory=True)
    return data_loader