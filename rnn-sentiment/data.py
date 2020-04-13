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

        pyplot.bar(range(0, len(loss_by_category)), loss_by_category, color='blue')
        pyplot.show()
        pyplot.savefig(fig_path)
        pyplot.clf()


if __name__ == "__main__":
    raise NotImplementedError
