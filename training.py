'''
Clasification of Parkinsons
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import argparse
import numpy as np
import random
import os
import tqdm
import pandas as pd
from collections import OrderedDict

# Softmax cross entropy
softmax = nn.Softmax(dim=1)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataListPath, namelist):
        self.dataDir = os.path.dirname(dataListPath)
        self.namelist = namelist

        # Input name list
        with open(dataListPath) as f:
            all_line = f.readlines()
            data = [line.replace("\n","").split(",") for line in all_line]

        data_del_row = np.delete(np.array(data), obj=0, axis=0) # delete 0 row
        data_input = np.delete(data_del_row, obj=[0, 17], axis=1) # delete 0&17 colums
        data_target = data_del_row[:,17]

        self.input = Subset(data_input, namelist)
        self.target = Subset(data_target, namelist)

    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, i):
        input = torch.from_numpy(self.input[i].astype(np.float32))
        target = torch.from_numpy(np.array(self.target[i], dtype=np.int32)).long()
        return input, target

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(22, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class AverageMeter(object): # ref https://github.com/4uiiurz1/pytorch-nested-unet/blob/master/train.py
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_args():
    parser = argparse.ArgumentParser(
        description='torch line drawing colorization')   
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=str, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', '-d', default='{}/parkinsons.data'.format(os.path.dirname(os.path.abspath(__file__))),
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='{}/output'.format(os.path.dirname(os.path.abspath(__file__))),
                        help='Directory to output the result')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    return args

def Make_list(fold_num, dataset):

    dataset_1, dataset_2, dataset_3, dataset_4 = dataset

    if fold_num == 0: # for debug
        train_list = np.concatenate((dataset_1, dataset_2, dataset_3, dataset_4))
        test1_list = np.concatenate((dataset_1, dataset_2))
        test2_list = np.concatenate((dataset_3, dataset_4))
    elif fold_num == 1: # fold1
        train_list = np.concatenate((dataset_1, dataset_2))
        test1_list = dataset_3
        test2_list = dataset_4
    elif fold_num == 2: # fold2
        train_list = np.concatenate((dataset_3, dataset_4))
        test1_list = dataset_1
        test2_list = dataset_2
    else:
        "Fold Number Error"
   
    return train_list, test1_list, test2_list

def training(args, train_loader, model, criterion, optimizer):

    model.train()

    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            losses.update(loss.item(), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
    ])

    return log

def test(args, test_loader, model):
    # switch to evaluate mode
    model.eval()

    log = []
    with torch.no_grad():
        for input, target in test_loader:
            input = input.cuda()

            # compute output
            output = model(input)
            x = softmax(output).max(1)[1].data.cpu().numpy()
            
            for batch in range(len(target)):
                mini = []
                mini.append(target.numpy()[batch])
                mini.append(x[batch])
                log.append(mini)

    return log

def main():
    args = parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# epoch: {}'.format(args.epoch))

    CDir = os.path.dirname(os.path.abspath(__file__))
    dataListPath = os.path.join(CDir, args.dataset)
    out_path = os.path.join(CDir, 'result', args.out)
    if not os.path.isdir(out_path):
        os.makedirs(out_path)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    # Added for reproducibility
    cudnn.deterministic = True
    print("Deterministic cuDNN is",cudnn.deterministic)

    # 2-fold cross validation
    index = list(range(0, 195))
    dataset_1, dataset_2, dataset_3, dataset_4 = torch.utils.data.random_split(index, [49, 49, 49, 48])
    dataset = (dataset_1, dataset_2, dataset_3, dataset_4)

    fold_size = 2
    accuracy = []
    for fold_num in range(fold_size):
        if not os.path.isdir('{}/fold_{}'.format(out_path, fold_num + 1)):
            os.makedirs('{}/fold_{}'.format(out_path, fold_num + 1))

        # Dataset
        print("------Loading the Dataset------")
        train_list, test1_list, test2_list = Make_list(fold_num + 1, dataset)

        train = Dataset(dataListPath, train_list)
        test1 = Dataset(dataListPath, test1_list)
        test2 = Dataset(dataListPath, test2_list)

        print("------Loaded the Dataset------")
        print("train_data num : ",train.__len__())
        print("test1_data num : ",test1.__len__())
        print("test2_data num : ",test2.__len__())

        # Iterator
        train_loader = DataLoader(train, train.__len__(), shuffle=True)
        test1_loader = DataLoader(test1, test1.__len__(), shuffle=False)
        test2_loader = DataLoader(test2, test2.__len__(), shuffle=False)

        # model
        net = Net()
        net.to(device)

        # define loss function and optimier
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(),lr=0.001)

        best_val1 = 1
        best_val2 = 1
        best_epoch1 = 1
        best_epoch2 = 1
        # train
        for epoch in range(args.epoch):
            training(args, train_loader, net, criterion, optimizer)
            val1_log = validate(args, test1_loader, net, criterion)
            val2_log = validate(args, test2_loader, net, criterion)

            if val1_log['loss'] < best_val1:
                best_val1 = val1_log['loss']
                best_epoch1 = epoch + 1

            if val2_log['loss'] < best_val2:
                best_val2 = val2_log['loss']
                best_epoch2 = epoch + 1

            torch.save(net.state_dict(), '{}/fold_{}/epoch_{}.npz'.format(out_path, fold_num + 1, epoch + 1))

            torch.cuda.empty_cache()

            # test
            if fold_num == 0:
                net.load_state_dict(torch.load('{}/fold_{}/epoch_{}.npz'.format(out_path, 1, best_epoch2)))
                test1_log = test(args, test1_loader, net)

                net.load_state_dict(torch.load('{}/fold_{}/epoch_{}.npz'.format(out_path, 1, best_epoch1)))
                test2_log = test(args, test2_loader, net)
            else:
                net.load_state_dict(torch.load('{}/fold_{}/epoch_{}.npz'.format(out_path, 2, best_epoch2)))
                test3_log = test(args, test1_loader, net)

                net.load_state_dict(torch.load('{}/fold_{}/epoch_{}.npz'.format(out_path, 2, best_epoch1)))
                test4_log = test(args, test2_loader, net)

    # accuracy
    log = np.concatenate((test1_log, test2_log, test3_log, test4_log), axis=0)
    count = 0
    for i in range(len(log)):
        if log[i,0] == log[i,1]:
            count += 1
    accu = count / int(len(log))
    accuracy.append(accu)
    print(accu)

    log = pd.DataFrame(log, columns=['GT', 'predict'])
    log.to_csv('{}/test_log_{}.csv'.format(out_path, seed), index=False)

    accuracy = pd.DataFrame(accuracy, columns=['Accuracy'])
    accuracy.to_csv('{}/accuracy.csv'.format(out_path), index=False)

if __name__ == '__main__':
    main()