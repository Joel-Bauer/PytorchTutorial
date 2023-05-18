import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) # assumes a n,c,x,y organisation with c = 1
        return F.softmax(self.fc1(x), dim=1)
