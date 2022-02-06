import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from datasets import Satellite_Dataset
from model import BP_2Net
import torch.nn.functional as F


def detect():
    n_feature = opt.n_feature
    #n_hidden = opt.n_hidden
    n_output = opt.n_output

    test_data = Satellite_Dataset(data_dir=opt.test_data,label_dir=opt.test_label, transform=opt.transform)
    test_loader = DataLoader(dataset=test_data, batch_size=opt.batch_size)


    net = BP_2Net(n_feature,n_output)
    net.load_state_dict(torch.load(opt.path_state_dict))
    net.eval()
    loss_mean = 0.
    correct = 0.
    total = 0.
    for i, data in enumerate(test_loader):
        # forward
        inputs, labels = data
        outputs = net(inputs)
        predicted = torch.max(F.softmax(outputs), 1)[1]
        intenion = 0 if predicted.numpy()[0] == 0 else 1
        print("模型获得{}元".format(intenion))

        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()
    print("Inference: Accuracy:{:.2%}".format(correct / total))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='./data.csv')
    parser.add_argument('--test_label', type=str, default='./label.csv')
    parser.add_argument('--path_state_dict', type=str, default='./weight.pt')
    parser.add_argument('--transform', type=bool, default=True)
    parser.add_argument('--n_feature', type=int, default=2)
    parser.add_argument('--n_output', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1)  # effective bs = batch_size * accumulate = 16 * 4 = 64

    opt = parser.parse_args()
    detect()