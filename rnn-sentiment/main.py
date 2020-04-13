import torch
import torchtext

from data import BatchStatistics, Evaluation
from model import SentRNN

def train(model, device, loader, batch_size, lr, optimizer):

    epochs = 30
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr, )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr, )
    display_inverse_freq = len(loader) + 1

    model.to(device)
    for p in model.parameters():
        p.to(device)

    all_batch_stats = []
    loader = iter(loader)
    img = next(loader)
    for e in range(epochs):
        batch_iter = 0
        while img:

            label = img.label
            img = img.text

            label = label - 1
            label = torch.nn.functional.one_hot(label, num_classes=5) 
            label = label.unsqueeze(0)

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
                #print("progress", (e/epochs) * 100 + batch_iter / len(loader) / epochs * 100, "%")
            batch_iter +=1
            img = next(loader)

    return model, all_batch_stats

def test(model, device, loader):

    batch_size = 64

    all_batch_stats = []
    for img, label in loader:
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

    text = torchtext.data.Field()
    label = torchtext.data.Field(sequential=False)

    data_train, data_val, data_test = torchtext.datasets.SST.splits(text, label,
                              fine_grained=True, train_subtrees=False)

    text.build_vocab(data_train, vectors=torchtext.vocab.Vectors(name="vector.txt", cache="./data"))
    label.build_vocab(data_train)

    iter_train, iter_val, iter_test = torchtext.data.BucketIterator.splits(
        (data_train, data_val, data_test), batch_size = 64)

    model = SentRNN(embeddings=text.vocab.vectors)
    model, stats = train(model, device, iter_train, 64, lr=0.01, optimizer="SGD")

    analysis = Evaluation(10, 1, stats)
    file_prefix = ""
    analysis.plot_loss(file_prefix + "train_loss.png")
    analysis.plot_accu(file_prefix + "train_accu.png")
    analysis.calc_loss_by_category()
    stats = test(model, device, iter_test)
    analysis = Evaluation(10, 1, stats)
    analysis.calc_loss_by_category()
    analysis.plot_loss_by_category(file_prefix + "test.png")
    analysis.print_summary()

if __name__ == "__main__":
    main()
