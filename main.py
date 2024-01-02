from data_loaders import train_loader, test_loader
from train import create_model, train_model
import torch

if __name__ == "__main__":
    model,criterion,optim = create_model()
    train_loss, test_loss, train_acc, test_loss, best_model, model = train_model(model,criterion,optim,30,train_loader,test_loader)
    print("Best Model Acc: {}".format(best_model["acc"]))
    print("Best Model: {}".format(best_model["model"]))
    torch.save(model.state_dict(), "models/potato_cnn.pth")