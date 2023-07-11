from torch import nn
import torch

# the transformer network
class NACCModel(nn.Module):

    def __init__(self, num_features, num_classes, nhead=2, nlayers=4, hidden=128):
        # call early initializers
        super(NACCModel, self).__init__()

        # we have a seperate, tiny transformer hidden dimension
        # mostly used as a sequential data squishifiaction layer
        D_transformer = 8

        # the entry network ("linear embedding")
        # bigger than 80 means that its going to be out of bounds and therefore
        # be masked out; so hard code 81
        self.linear0 = nn.Linear(1, D_transformer)
        
        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=D_transformer, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # flatten!
        self.flatten = nn.Flatten()

        # dropoutp!
        self.dropout = nn.Dropout(0.4)

        # prediction network
        self.linear1 = nn.Linear(D_transformer*num_features, hidden)
        self.norm = nn.BatchNorm1d(hidden)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, hidden)
        self.linear4 = nn.Linear(hidden, num_classes)
        self.softmax = nn.Softmax(1)

        # loss
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, mask, labels=None):

        net = self.linear0(torch.unsqueeze(x, dim=2))
        # recall transformers are seq first
        net = self.encoder(net.transpose(0,1), src_key_padding_mask=mask).transpose(0,1)
        net = self.flatten(net)
        net = self.dropout(net)
        net = self.linear1(net)
        net = self.norm(net)
        net = self.gelu(net)
        net = self.dropout(net)
        net = self.gelu(self.linear2(net))
        net = self.gelu(self.linear3(net))
        net = self.linear4(net)
        net = self.dropout(self.softmax(net))

        loss = None
        if labels is not None:
            # TODO put weight on MCI
            # loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])
            loss = self.cross_entropy(net, labels)

        return { "logits": net, "loss": loss }

