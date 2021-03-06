import os
import argparse
import torch
import json
import copy
import math
import numpy as np
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import random
import model as mdl
from torch.multiprocessing import Process

from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import timeit


device = "cpu"
torch.set_num_threads(4)
batch_size = 256 # batch for one node
randomseed = 1234
np.random.seed(randomseed)

def init_process(rank, size, fn, master_ip, outputfile, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = '6585'
    dist.init_process_group(
        backend,
        init_method='tcp://10.10.1.1:6585',
        rank=rank,
        world_size=size)
    fn(rank, size, outputfile)


def test_model(model, test_loader, criterion, outputfile):
    fp = open(outputfile+"_r"+str(dist.get_rank())+"_size"+str(dist.get_world_size()), "a")

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    fp.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    fp.close()


""" Dataset partitioning helper """
class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=randomseed):
        self.data = data
        self.partitions = []
        rng = random.Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


""" Partitioning CIFAR10 """
def partition_dataset(normalize):
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
    dataset = datasets.CIFAR10(root="./data", train=True,
                                download=True, transform=transform_train)
    size = dist.get_world_size()
    bsz = int(batch_size / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                         batch_size=bsz,
                                         num_workers=2,
                                         sampler=None,
                                         shuffle=True,
                                         pin_memory=True)
    return train_set, bsz


def run(rank, size, outputfile):
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    """set randomseed"""
    torch.manual_seed(randomseed)
    print('manual_seed=',randomseed)

    """set up data"""
    train_set, bsz = partition_dataset(normalize)

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    test_set = datasets.CIFAR10(root="./data", train=False,
                                download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              num_workers=2,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    """set up model"""
    model = mdl.VGG11()
    model.to(device)
    model = DDP(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1,
                          momentum=0.9, weight_decay=0.0001)

    num_batches = math.ceil(len(train_set.dataset) / float(bsz))

    """write output to file"""
    if os.path.exists(outputfile):
        os.remove(outputfile)
    fp = open(outputfile+"_r"+str(dist.get_rank())+"_size"+str(dist.get_world_size()), "a")

    """start training"""
    total_epoch = 1
    for epoch in range(total_epoch):
        running_loss = 0.0
        # remember to exit the train loop at end of the epoch
        for batch_idx, (data, target) in enumerate(train_set):
            if batch_idx < 10:
                start = timeit.default_timer()

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if batch_idx % 20 == 19:  # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 20))
                fp.write('[%d, %5d] loss: %.3f\n' % (epoch + 1, batch_idx + 1, running_loss / 20))
                running_loss = 0.0

            if batch_idx == 0:
                fp.write("Batch\trunning time\n")
            if batch_idx < 10:
                end = timeit.default_timer() - start
                print("Batch "+str(batch_idx)+" running time:"+str(end))
                fp.write('%d\t%.5f\n' % (batch_idx, end) )

        # # training stop
        fp.close()


    test_model(model, test_loader, criterion, outputfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--master_ip", type=str, default='10.10.1.1')
    parser.add_argument("--outputfile", type=str, default='output_p2a.txt')
    args = parser.parse_args()

    processes = []
    p = Process(target=init_process, args=(args.rank, args.num_nodes, run, args.master_ip, args.outputfile))
    p.start()
    processes.append(p)

    p.join()
