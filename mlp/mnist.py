import torch
import torchvision
import numpy as np

from dataclasses import dataclass
import matplotlib.pyplot as pyplot

@dataclass
class BatchStatistics:
    #todo. keep tensors or convert all to list at first
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
    num_epochs       : int
    batch_statistics : list

    def plot_loss(self, fig_path):
        losses = [batch.loss_sum.item() for batch in self.batch_statistics]
        losses = [sum([bloss for bloss in epoch])/len(epoch)
                 for epoch in [losses[i: i + len(losses)//self.num_epochs] for i in range(0, len(losses), len(losses)//self.num_epochs)]]

        pyplot.plot(losses)
        pyplot.show()
        pyplot.savefig(fig_path)
        pyplot.clf()

    def plot_accu(self, fig_path):
        accus = [batch.accu_mean for batch in self.batch_statistics]
        accus = [sum([baccu for baccu in epoch])/len(epoch)
                 for epoch in [accus[i: i + len(accus)//self.num_epochs] for i in range(0, len(accus), len(accus)//self.num_epochs)]]
        pyplot.plot(accus)
        pyplot.show()
        pyplot.savefig(fig_path)
        pyplot.clf()

    def print_summary(self):
        #redundant .item. 
        loss_epochs = [batch.loss_sum.item()for batch in self.batch_statistics]
        loss_total = sum(loss_epochs) / len(loss_epochs)
        accu_epochs = [batch.accu_mean for batch in self.batch_statistics]
        accu_total = sum(accu_epochs) / len(loss_epochs)
        print("loss:", loss_total)
        print("accu:", accu_total)

    def calc_loss_by_category(self):
        correct_by_category   = [0 for _ in range(self.categories_n)]
        samples_n_by_category = [0 for _ in range(self.categories_n)]
        for bs in self.batch_statistics:
            for i, (pred, label) in enumerate(zip(bs.all_preds, bs.all_labels)):
                samples_n_by_category[label.item()] += 1
                if torch.argmax(pred, dim=-1).item() == label.item():
                    correct_by_category[label.item()] += 1 

        return [score/all for (score, all) in zip(correct_by_category, samples_n_by_category)]

    def plot_loss_by_category(self, fig_path, loss_by_category=None):

        if not loss_by_category:
            loss_by_category = self.calc_loss_by_category()

        pyplot.clf()
        pyplot.bar(range(0, len(loss_by_category)), loss_by_category, color='blue')
        pyplot.show()
        pyplot.savefig(fig_path)      


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear = torch.nn.Linear(28 * 28, 10)
    
    def forward(self, x, label):
        #view ?
        x = x.reshape(-1, 28* 28)
        x = self.linear(x)
        label = torch.nn.functional.one_hot(label, 10).float()
        loss  = torch.nn.functional.binary_cross_entropy_with_logits(x, label)
        return loss.mean(dim=0), x


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
        return loss.mean(dim=0), x


def train(device):

    #model = ConvNet()
    model = MLP()
    batch_size = 64
    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=True, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)

    lr = 0.001 
    epochs = 50
    optimizer = torch.optim.SGD(model.parameters(), lr, )
    display_inverse_freq = len(dataloader) + 1

    model.to(device)
    for p in model.parameters():
        p.to(device)

    all_batch_stats = []
    for e in range(epochs):
        batch_iter = 0
        for img, label in dataloader:
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss, p = model.forward(img, label)
            loss.backward()
            optimizer.step()
            #could use a normal class
            bs = BatchStatistics(batch_size, loss, torch.argmax(p, dim=1) == label,
                                 (torch.argmax(p, dim=1) == label).sum().item()/batch_size,
                                 p, label, epoch=e)
            all_batch_stats.append(bs)

            if batch_iter % display_inverse_freq == 0:
                print("batch loss", bs.loss_sum.item())
                print("batch acc", bs.accu_mean, "%")
                print("progress", (e/epochs) * 100 + batch_iter / len(dataloader) / epochs * 100, "%")
            batch_iter +=1

    return model, all_batch_stats

def test(model, device):

    batch_size = 64
    dataloader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=False, download=True,
                                transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))])),
        batch_size=batch_size, shuffle=True)

    all_batch_stats = []
    for img, label in dataloader:
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            loss, p = model.forward(img, label)
        bs = BatchStatistics(batch_size, loss, torch.argmax(p, dim=1) == label,
                                 (torch.argmax(p, dim=1) == label).sum().item()/batch_size,
                                 p, label, epoch=0)
        all_batch_stats.append(bs)

    return all_batch_stats

def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    model, stats = train(device)
    analysis = Evaluation(10, 10, stats)
    analysis.plot_loss("train_loss.png")
    analysis.plot_accu("train_accu.png")
    analysis.calc_loss_by_category()
    stats = test(model, device)
    analysis = Evaluation(10, 1, stats)
    analysis.calc_loss_by_category()
    analysis.plot_loss_by_category("test.png")
    analysis.print_summary()

if __name__ == "__main__":
    main()