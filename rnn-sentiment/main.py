import torch
import torchtext

from data import BatchStatistics, Evaluation
from model import SentRNN

def train(model, device, loader, batch_size, lr, optimizer):

    epochs = 10
    if optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr, weight_decay=10**-7)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=10**-7)
    display_inverse_freq = len(loader) + 1

    model.to(device)
    for p in model.parameters():
        p.to(device)

    all_batch_stats = []

    grad_clip_limit = 5

    #loader = torch.nn.utils.rnn.pad_sequence(loader)
    #loader = torch.nn.utils.rnn.pack_padded_sequence(loader)
    for e in range(epochs):
        batch_iter = 0
        for img in loader:
            label = img.label
            img = img.text

            label = label - 1
            label = torch.nn.functional.one_hot(label, num_classes=5) 
            label = label.unsqueeze(1)

            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            loss, p = model.forward(img, label)

            loss.backward()
            optimizer.step()
            #could use a normal class
            #print(p.shape)
            #print(label.shape)
            label = label.squeeze(1)
            bs = BatchStatistics(batch_size, loss, torch.argmax(p, dim=1) == torch.argmax(label, dim=1),
                                 (torch.argmax(p, dim=1) == torch.argmax(label, dim=1)).sum().item()/batch_size,
                                 p, torch.argmax(label, dim=1), epoch=e)
            all_batch_stats.append(bs)

            if batch_iter % display_inverse_freq == 0:
                print("batch loss", bs.loss_sum.item())
                print("batch acc", bs.accu_mean * 100, "%")
                #print("progress", (e/epochs) * 100 + batch_iter / len(loader) / epochs * 100, "%")
            batch_iter +=1
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_limit)

    return model, all_batch_stats

def test(model, device, loader):

    batch_size = 32

    all_batch_stats = []
    for img in loader:
        label = img.label
        img = img.text

        if img.shape[0] != batch_size:
            print("Skipping batch with unexpected size")
            continue

        label = label - 1
        label = torch.nn.functional.one_hot(label, num_classes=5) 
        label = label.unsqueeze(1)
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            loss, p = model.forward(img, label)
        label = label.squeeze(1)
        bs = BatchStatistics(batch_size, loss, torch.argmax(p, dim=1) == torch.argmax(label, dim=1),
                                 (torch.argmax(p, dim=1) == torch.argmax(label, dim=1)).sum().item()/batch_size,
                                 p, torch.argmax(label, dim=1), epoch=0)
        all_batch_stats.append(bs)

    return all_batch_stats

def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"

    text = torchtext.data.Field(batch_first=True)
    label = torchtext.data.Field(sequential=False, batch_first=True)

    data_train, data_val, data_test = torchtext.datasets.SST.splits(text, label,
                              fine_grained=True, train_subtrees=False)

    text.build_vocab(data_train, vectors=torchtext.vocab.Vectors(name="vector.txt", cache="./data"))
    label.build_vocab(data_train)

    iter_train, iter_val, iter_test = torchtext.data.BucketIterator.splits(
        (data_train, data_val, data_test), batch_size = 32)

    model = SentRNN(embeddings=text.vocab.vectors)
    model, stats = train(model, device, iter_train, 32, lr=0.0001, optimizer="Adam")

    analysis = Evaluation(5, 1, stats)
    file_prefix = ""
    analysis.plot_loss(file_prefix + "train_loss.png")
    analysis.plot_accu(file_prefix + "train_accu.png")
    analysis.calc_loss_by_category()
    stats = test(model, device, iter_test)
    analysis = Evaluation(5, 1, stats)
    analysis.calc_loss_by_category()
    analysis.plot_loss_by_category(file_prefix + "test.png")
    analysis.print_summary()

if __name__ == "__main__":
    main()
