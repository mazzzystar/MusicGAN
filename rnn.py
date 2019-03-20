from wavegan import *
from util import *
cuda = True if torch.cuda.is_available() else False
from util import pass_through_rnn_g


class RNN(nn.Module):
    """
    * Put z as the only input of every GRU node, same vs. different z for every GRU node.
    """
    def __init__(self, input_size, hidden_size, batch_size, latent_dim=128, model="gru", n_layer=1):
        super(RNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_layers = n_layer

        self.rnn = nn.GRU(input_size, hidden_size, n_layer, batch_first=True)

    def forward(self, z, hidden):
        _output, hidden = self.rnn(z, hidden)
        return _output, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        if cuda:
            hidden = hidden.cuda()
        return Variable(hidden)


class RNN2(nn.Module):
    """
    * Except the first input z, each input of the RNN is the output of previous RNN.
    * Every previous output concat current z to be the new input
    """
    def __init__(self, input_size, hidden_size, batch_size, model="gru", n_layer=1):
        super(RNN2, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layer

        self.rnn = nn.GRU(input_size, hidden_size, n_layer, batch_first=True)

    def forward(self, z, hidden):
        """
        Here is the difference: z will only be feed into the first node.

        z: (G_NUMS, batch, input_size)
        """
        outputs = []

        assert z.shape[1] == G_NUMS

        _output = None
        for i in range(G_NUMS):
            if _output is None:
                _output = z[:, i, :].unsqueeze(1)
            else:
                _output = _output + z[:, i, :].unsqueeze(1)
            # print("shape:={}".format(_output.shape))
            _output, hidden = self.rnn(_output, hidden)
            outputs.append(_output)
        return outputs, hidden

    def init_hidden(self):
        hidden = torch.zeros(self.n_layers, self.batch_size, self.hidden_size)
        if cuda:
            hidden = hidden.cuda()
        return Variable(hidden)

"""
start = time.time()
x = Variable(torch.randn(2, G_NUMS, 128))
model = RNN2(x.shape[-1], hidden_size=x.shape[-1], batch_size=x.shape[0])
netG = WaveGANGenerator(verbose=True)
hidden = model.init_hidden()
gen_lst, hidden = pass_through_rnn_g(x, hidden, same_z=False, one_z=False, rnnModel=model, netG=netG, batch_size=2)
fake = batch_fake_generator(gen_lst, 4)
print(fake.shape)
cur = time.time()
print("time used:{}".format(cur - start))

kernel_size = 25
pad = math.ceil(float(kernel_size-4) / 2)
print("pad={}".format(pad))
D = WaveGANDiscriminator(verbose=True)
out2 = D(fake)
print(out2.shape)
print("time used:{}".format(cur - start))
"""