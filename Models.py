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


# A model with 2 convolutional layers and 1 hidden fully connected layer
class ConvNet1(nn.Module):
    def __init__(self, dropout_rate=0.05):
        
        super(ConvNet1, self).__init__()

        self.insize = (1,1,28,28) # input dimensions of the image (m,c,h,w)

        self.Ech1, self.Ech2 = 3, 12 # define the number of channels for the conv layers
        self.Ek1, self.Ek2 = (3, 3), (3, 3)  # define 2d kernel size
        self.Es1, self.Es2 = (1, 1), (1, 1)  # define 2d strides
        self.Ep1, self.Ep2 = (1, 1), (1, 1)  # define 2d padding
        self.MP1, self.MP2 = 2, 2 # define max pooling (downscaling) 

        self.convoutsize = self.Ech2* (self.insize[-1]//self.MP1//self.MP2)**2

        
        self.fc_hidden1 = 100 # number of hidden units
        self.outsize = 10 # number of output units, categories

        self.conv_layers = nn.Sequential(                            
            nn.Conv2d(1, self.Ech1, self.Ek1, stride=self.Es1, padding=self.Ep1),           ## conv layer 1         
            nn.BatchNorm2d(self.Ech1, momentum=0.01),                                           # normalizes the output values to reduce the chance of vanishing or exploding gradients
            nn.ReLU(),                                                                          # nonlinearity
            nn.MaxPool2d(self.MP1),                                                             # reduces the h and w of the output                           
            nn.Conv2d(self.Ech1, self.Ech2, self.Ek2, stride=self.Es2, padding=self.Ep2),   ## conv layer 2                
            nn.BatchNorm2d(self.Ech2, momentum=0.01),
            nn.ReLU(),
			nn.MaxPool2d(self.MP2)
        )

        # self.fc1_drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(self.convoutsize, self.fc_hidden1)
        
        self.fc_out = nn.Linear(self.fc_hidden1,self.outsize)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        # x = self.fc1_drop(x)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc_out(x), dim=1)
    