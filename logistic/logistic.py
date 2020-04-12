import mnist_data_loader
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Statistics:
    n_data_train: int
    n_data_test: int

    hist_loss = []
    hist_acc = []
    epoch_loss = []
    pred_train_n_true = 0
    pred_test_n_true = 0

    def sum_round(self) -> None:
        self.hist_loss.append(float(np.mean(self.epoch_loss)))
        self.hist_acc.append(float(self.pred_train_n_true/self.n_data_train))
        self.epoch_loss = []
        self.pred_train_n_true = 0
        #print(self.hist_loss.shape, self.hist_acc.shape)
    def get_acc_test(self) -> float:
        return self.pred_test_n_true/self.n_data_test


mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=False)
# training dataset
train_set = mnist_dataset.train
# test dataset
test_set = mnist_dataset.test
print("Training dataset size:" , train_set.num_examples)
print("Test dataset size:" , test_set.num_examples)

import matplotlib.pyplot as plt
example_id = 0
image = train_set.images[example_id] # shape = 784 (28*28)
label = train_set.labels[example_id] # shape = 1

#plt.imshow(np.reshape(image,[28,28]),cmap="gray")
#plt.show()

def weight_init(img_shape):
    return np.zeros(img_shape)

def sigmoid(z):
    return 1 / (1 + math.e**(-z))

def predict(img, weight):
    pred = np.dot(img, weight)
    return(sigmoid(pred))

def cost(weight, pred, img, label):
    #assert (label in [0, 1])

    err = - 1/weight.shape[0] * label * np.log2(pred) + (1 - label) * np.log2(1 - pred) 
    return err

def grad(weight, err, img, label):
    g = 1/ weight.shape[0] * np.dot(img.T, err) # /weight.shape[0] #double divide?
    return g.flatten()

try:
    batch_size = int(sys.argv[1])
    max_epoch = int(sys.argv[2])
except ValueError as e:
    print("use: 'batch size' 'epochs'")
    raise ValueError("use: 'batch size' 'epochs'")

weight = np.zeros(image.shape)

stats = Statistics(int(train_set.images.shape[0]), int(test_set.images.shape[0]))

for epoch in range(0, max_epoch):
    print("epoch:", epoch)
    iter_per_batch = train_set.num_examples // batch_size
    for batch_id in range(0, iter_per_batch):
        # get the data of next minibatch (have been shuffled)
        batch = train_set.next_batch(batch_size)
        inp, label = batch
        # Convert input label
        label = np.array([0 if l == 3 else 1 for l in label])
        # prediction
        pred = predict(inp, weight)
        # calculate the loss (and accuracy)
        loss = cost(weight, pred, inp, label)
        # update weights
        g = grad(weight, loss, inp, label)
        weight = weight + g
        # update statistics
        stats.epoch_loss += (list(- 100 * loss))
        pred = [1 if p >= 0.5 else 0 for p in pred]
        stats.pred_train_n_true += (int(np.sum([pred == label])))
    stats.sum_round()

n_correct = 0
iter_per_batch = test_set.num_examples // batch_size
for batch_id in range(0, iter_per_batch):
    batch = test_set.next_batch(batch_size)
    inp, label = batch
    #[0 if l == 3 else 1 for l in label]
    pred = predict(inp, weight)
    pred = [3 if p < 0.5 else 6 for p in pred]
    print(pred == label)
    stats.pred_test_n_true += np.sum(pred == label)


def plot_hist(stats):
    print(stats.hist_acc)
    print(stats.hist_loss)
    plt.plot([i for i in range(len(stats.hist_acc))], stats.hist_acc)
    plt.plot([i for i in range(len(stats.hist_loss))], stats.hist_loss)
    plt.xlabel('epoch')
    plt.ylabel('loss / accuracy ')
    plt.show()

plot_hist(stats)

print("Accuracy is", stats.pred_test_n_true, "/", stats.n_data_test,
      100*stats.pred_test_n_true/stats.n_data_test, "%")
