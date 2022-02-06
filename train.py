import torch
import argparse
from model import BP_2Net
import torch.nn as nn
import torch.optim as optim
from datasets import Satellite_Dataset
from model import BP_2Net
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train():
    n_feature = opt.n_feature
    #n_hidden = opt.n_hidden
    n_output = opt.n_output

    # 构建MyDataset实例
    train_data = Satellite_Dataset(data_dir=opt.data_dir,label_dir=opt.label_dir, transform=opt.transform)
    #valid_data = RMBDataset(data_dir=opt.valid_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True)
    #valid_loader = DataLoader(dataset=valid_data, batch_size=opt.batch_size)

    # ============================ step 2/5 模型 ============================
    net = BP_2Net(n_feature,n_output)
    net.initialize_weights()

    # ============================ step 3/5 损失函数 ============================
    criterion = nn.CrossEntropyLoss()  # 选择损失函数

    # ============================ step 4/5 优化器 ============================
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9)  # 选择优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # 设置学习率下降策略

    # ============================ step 5/5 训练 ============================
    train_curve = list()
    valid_curve = list()
    Acc = []

    for epoch in range(opt.epochs):
        acc_temp = 0
        loss_mean = 0.
        correct = 0.
        total = 0.

        net.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            outputs = net(inputs)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            predicted = torch.max(F.softmax(outputs), 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().numpy()
            if correct / total > acc_temp:
                acc_temp = correct / total
            # 打印训练信息
            loss_mean += loss.item()
            train_curve.append(loss.item())
            if (i + 1) % opt.log_interval == 0:
                loss_mean = loss_mean / opt.log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, opt.epochs, i + 1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.
        Acc.append(acc_temp)
        scheduler.step()  # 更新学习率
        if epoch > 1 and Acc[epoch] > Acc[epoch-1]:
            net_state_dict = net.state_dict()
            torch.save(net_state_dict, opt.save_model_path)
            print("Model already save")
        else:
            continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch_size', type=int, default=10)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--lr', type=float, default=0.02)
    parser.add_argument('--data_dir', type=str, default="./data.csv")
    parser.add_argument('--label_dir', type=str, default="./label.csv")
    parser.add_argument('--save_model_path', type=str, default="./weight.pt")
    parser.add_argument('--n_feature', type=int, default=2)
    #parser.add_argument('--n_hidden', type=int, default=10)
    parser.add_argument('--n_output', type=int, default=2)
    parser.add_argument('--transform', type=bool, default=True)
    parser.add_argument('--log_interval', type=int, default=20)



    opt = parser.parse_args()
    train()