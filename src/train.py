import torch
from network import CNN
import torch.nn as nn
from copy import deepcopy
import numpy as np

def create_model():
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=.001,weight_decay=.0001)
    return model,criterion,optim

def train_model(model,criterion,optim,epochs,train_loader, test_loader):
    best_model = {"acc":0, "model": None}
    train_acc = []
    test_acc = []
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        print("\nEpoch: {}".format(epoch+1))
        batch_acc = []
        batch_loss = []
        model.train()
        for X,y in train_loader:
            yhat = model(X)
            loss = criterion(yhat, y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            batch_loss.append(loss.detach())
            matches = (torch.argmax(yhat,axis=1)==y).float()
            acc = 100*torch.mean(matches)
            batch_acc.append(acc)
        
        train_loss.append(np.mean(np.mean(batch_loss)))
        train_acc.append(np.mean(batch_acc))

        model.eval()
        X,y = next(iter(test_loader))
        with torch.no_grad():
            yhat = model(X)
            loss = criterion(yhat, y)
        test_loss.append(loss.detach())
        test_acc.append(100*torch.mean((torch.argmax(yhat,axis=1)==y).float()))

        if test_acc[-1]>best_model["acc"]:
            best_model["acc"] = test_acc[-1]
            best_model["model"] = deepcopy(model.state_dict())
        
        print("\nSUMMARY: Epoch: {}, Train Acc: {}, Test_Acc:{}".format(epoch+1,train_acc[-1],test_acc[-1]))
    
    return train_loss, test_loss, train_acc, test_acc, best_model, model

