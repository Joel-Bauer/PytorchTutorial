import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.fc1 = nn.Linear(784, 196)
        # self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(196, 49)
        # self.fc2_drop = nn.Dropout(0.2)
        self.fc3 = nn.Linear(49, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) # assumes a n,c,x,y organisation with c = 1
        x = F.relu(self.fc1(x))
        # x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        # x = self.fc2_drop(x)
        return F.softmax(self.fc3(x), dim=1)
