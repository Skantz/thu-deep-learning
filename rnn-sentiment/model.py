import torch

class SentRNN(torch.nn.Module):
    def __init__(self, inp_size=18280, h_size=300, out_size=5, embeddings=None):
        if embeddings == None:
            raise NotImplementedError
        super(SentRNN, self).__init__()
        self.inp_size = inp_size
        self.h_size = h_size
        self.out_size = out_size
        
        self.encoder = torch.nn.Embedding(inp_size, h_size)
        self.recurrent = torch.nn.GRU(h_size, h_size, num_layers=1)
        self.fconnected = torch.nn.Linear(h_size, out_size)

        self.encoder.weight.data.copy_(embeddings)

    def forward(self, x, label):

        x = self.encoder(x)
        _, h = self.recurrent(x)
        x = self.fconnected(h)
        label = label.double()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x, label)
        return loss.mean(dim=0), x

