import torch
import torchvision
import numpy as np

from dataclasses import dataclass
import matplotlib.pyplot as pyplot

@dataclass
class BatchStatistics:
    batch_size : int
    loss_sum   : float
    accu_list  : list
    accu_mean  : float
    all_preds  : list
    all_labels : list
    epoch      : int

@dataclass
class Evaluation:
    categories_n     : int
    batch_statistics : list
    loss_by_category_over_batches : list

    def plot_loss(self):
        losses = [batch.loss_sum for batch in self.batch_statistics]
        pyplot.plot(losses)
        pyplot.show()

    def calc_loss_by_category(self):
        correct_by_category   = [0 for _ in range(self.categories_n)]
        samples_n_by_category = [0 for _ in range(self.categories_n)]
        for bs in self.batch_statistics:
            for i, (pred, label) in enumerate(zip(bs.all_preds, bs.all_labels)):
                samples_n_by_category[label[i]] += 1
                if torch.argmax(pred[i], dim=1) == label[i]:
                    correct_by_category[label[i]] += 1 
        print(correct_by_category)


    def plot_loss_by_category(self):
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
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)

    lr = 0.01
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr, )
    display_inverse_freq = 500

    all_batch_stats = []
    for e in range(epochs):
        batch_iter = 0
        for img, label in dataloader:
            optimizer.zero_grad()
            loss, p = model.forward(img, label)
            loss.backward()
            optimizer.step()
            #couöd use a normal class
            bs = BatchStatistics(batch_size, loss, torch.argmax(p, dim=1) == label,
                                 (torch.argmax(p, dim=1) == label).sum().item()/batch_size,
                                 p, label, epoch=e)
            all_batch_stats.append(bs)
    
            if batch_iter % display_inverse_freq:
                print("batch loss", bs.loss_sum.item())
                print("batch acc", bs.accu_mean, "%")
            batch_iter +=1

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