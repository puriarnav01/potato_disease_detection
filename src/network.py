import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cblock1 = nn.Sequential(
            nn.Conv2d(3,10,kernel_size=5,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
            )
        
        self.cblock2 = nn.Sequential(
            nn.Conv2d(10,20,kernel_size=5,stride=1,padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )

        expected_size = 56180
        self.expected_size = expected_size
        self.fc1 = nn.Linear(expected_size,1000)
        self.fc2 = nn.Linear(1000,500)
        self.output = nn.Linear(500,7)

    def forward(self,x):
        #print(x.size())
        x = self.cblock1(x)
        #print(x.size())
        x = self.cblock2(x)
        #print(x.size())
        x = x.view(-1,self.expected_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

# X = torch.rand(1,3,225,225)
# model = CNN()
# yhat = model(X)