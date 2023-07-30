from torch import nn
import torch

# the transformer network
class NACCModel(nn.Module):

    def __init__(self, num_features, num_classes, nhead=4, nlayers=2, hidden=128):
        # call early initializers
        super(NACCModel, self).__init__()

        # the entry network ("linear embedding")
        # bigger than 80 means that its going to be out of bounds and therefore
        # be masked out; so hard code 81
        self.linear0 = nn.Linear(1, hidden)
        
        # the encoder network
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        # flatten!
        self.flatten = nn.Flatten()

        # dropoutp!
        self.dropout = nn.Dropout(0.3)

        # prediction network
        self.linear1 = nn.Linear(hidden*num_features, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, num_classes)

        self.gelu = nn.GELU()
        self.softmax = nn.Softmax(1)

        # loss
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, mask, labels=None):

        net = self.linear0(torch.unsqueeze(x, dim=2))
        # recall transformers are seq first
        net = self.encoder(net.transpose(0,1), src_key_padding_mask=mask).transpose(0,1)
        net = self.flatten(net)
        net = self.gelu(net)
        net = self.dropout(net)
        net = self.linear1(net)
        net = self.gelu(net)
        net = self.linear2(net)
        net = self.gelu(net)
        net = self.dropout(net)
        net = self.linear3(net)
        net = self.softmax(net)

        loss = None
        if labels is not None:
            # TODO put weight on MCI
            # loss = (torch.log(net)*labels)*torch.tensor([1,1.3,1,1])
            loss = self.cross_entropy(net, labels)

        return { "logits": net, "loss": loss }

