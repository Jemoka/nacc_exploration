import torch
import torch.nn as nn
import torch.nn.functional as F

class TraceModel(nn.Module):

    def __init__(self, hidden=256):
        super(TraceModel, self).__init__()

        self.l0 = nn.Linear(4, hidden)
        self.l1 = nn.Linear(hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x, labels=None):
        net = self.gelu(self.l0(x))
        net = self.gelu(self.l1(net))
        net = self.gelu(self.l2(net))
        net = torch.squeeze(self.sigmoid(self.l3(net)))

        if labels != None:
            loss = self.loss(net, labels.float())

            return {"output": net,
                    "predictions": net > 0.5,
                    "loss": loss}

        else:
            return {"output": net,
                    "predictions": net > 0.5}

