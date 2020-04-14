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
        self.recurrent = torch.nn.GRU(h_size, h_size, num_layers=1, batch_first=True)
        self.fconnected = torch.nn.Linear(h_size, out_size)

        self.encoder.weight.data.copy_(embeddings)
        self.hidden_state = torch.zeros(1, 32, h_size).to('cuda')

    def forward(self, x, label):
        self.hidden_state = torch.zeros(1, 32, 300).to('cuda')
        #print(x)
        x = self.encoder(x)
        #print("aftger", x)
        x, _ = self.recurrent(x, self.hidden_state) #self.hidden_state) #, self.hidden_state)
        #max pool
        #x, _ = torch.max(x, dim=1)
        x = x[:, -1, :]
        #pytorch transposes hidden.
        #print("x", x.shape)
        #print("h", self.hidden_state.shape) 
        x = self.fconnected(x) #torch.transpose(x, 0, 1))
        #x = x.transpose(0, 1)
        #label = label.double()
        x = torch.nn.functional.sigmoid(x)
        loss = torch.nn.functional.binary_cross_entropy(x, label.squeeze(1).float(), reduction="mean")
        lp = torch.nn.functional.softmax(x, dim=1)

        #self.hidden_state = self.hidden_state.detach()
        #self.recurrent.zero_grad()
        #x is not prob..
        return loss, x

