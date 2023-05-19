import torch.nn as nn
import torch.nn.functional as F

# A linear model
class LFF(nn.Module):
    def __init__(self):
        super(LFF, self).__init__()
        
        self.fc1 = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) # assumes a n,c,x,y organisation with c = 1
        return F.softmax(self.fc1(x), dim=1)

# A model with one small hidden layer 
class DFF_tiny(nn.Module):
    def __init__(self):
        super(DFF_tiny, self).__init__()
        
        self.fc1 = nn.Linear(784, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, 28*28) # assumes a n,c,x,y organisation with c = 1
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

# A model with 2 hidden layers
class DFF(nn.Module):
    def __init__(self):
        super(DFF, self).__init__()
        
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
