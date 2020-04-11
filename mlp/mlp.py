import torch
import torchvision
import numpy as np

from dataclasses import dataclass
from matplotlib import pyplot

@dataclass
class BatchStatistics:
    batch_size : int
    loss_sum   : float
    accu_list  : list
    accu_mean  : float
    def __init__(self, preds, label, loss):
        self.batch_size = len(label)
        self.loss_sum  = loss
        self.accu_list = torch.argmax(preds, dim=1) == label
        self.accu_mean = self.accu_list.sum().item()/self.batch_size

@dataclass
class Evaluation:
    train_statistics    : list
    test_statistics     : list

    def __init__(self):
        self.train_statistics = []
        self.test_statistics  = []

    def plot_loss(self):
        pass

    def calc_loss_by_category(self):
        pass

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
    
    def forward(self, x):
        pass


class ConvNet(torch.nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.cl1  = torch.nn.Conv2d(1,  32, kernel_size=2, stride=1)
        self.cl2  = torch.nn.Conv2d(32, 64, kernel_size=2, stride=1)
        self.fcl1 = torch.nn.Linear(43264, 400)
        self.fcl2 = torch.nn.Linear(400,   10)
        
    def forward(self, x, label):
        x = self.cl1(x)
        x = self.cl2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcl1(x)
        x = self.fcl2(x)
        label = torch.nn.functional.one_hot(label, 10).float()
        loss  = torch.nn.functional.binary_cross_entropy_with_logits(x, label)
        return loss, x

def train():

    model = ConvNet()
    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))])),
        batch_size=32, shuffle=True)

    lr = 0.01
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr, )

    total_loss = []
    for e in range(epochs):
        epoch_loss = []
        for img, label in dataloader:
            optimizer.zero_grad()
            loss, p = model.forward(img, label)
            loss.backward()
            optimizer.step()
            print("loss", loss)
            print(p.shape)
            print("argmax p", torch.argmax(p, dim=1))
            print("label", label)
            print((torch.argmax(p, dim=1) == label).sum().item()/len(label), "acc")
            epoch_loss.append(loss.item())
        sloss = sum(epoch_loss)/len(epoch_loss)
        print("loss in epoch e", e, " was ", sloss)
        total_loss.append(sloss)
    
    return model, stats

def test(model):

    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))])),
        batch_size=32, shuffle=True)

    stats = Evaluation()
    for img, label in dataloader:
        with torch.no_grad:
            loss, p = model.forward(img, label)



def main():
    model, stats = train()
    test(model)

if __name__ == "__main__":
    main()