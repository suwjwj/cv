import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from utils import *
from models import *

def train(model, train_loader, test_loader, loss_func, optimizer, epochs):
    train_losses = []
    test_losses = []
    test_acc = []
    min_loss = 1e10
    patience = 10  # 早停耐心值
    counter = 0    # 计数器
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        lss = 0.0
        # 添加进度显示
        print(f"Epoch [{epoch+1}/{epochs}]", end=" ")
        
        for i, (x, y) in enumerate(train_loader):
            out = model(x)
            y_ = torch.unsqueeze(y,dim=1).float()
            loss = loss_func(out, torch.unsqueeze(y,dim=1).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lss = loss.item()
            
            # 每隔一定步数显示当前loss
            if i % 10 == 0:
                print(f"\rEpoch [{epoch+1}/{epochs}] Batch [{i}/{len(train_loader)}] Loss: {lss:.4f}", end="")
        
        train_losses.append(lss)
        
        # 评估阶段
        model.eval()
        acc = 0
        num = 0
        test_loss = 0
        
        for i, (x, y) in enumerate(test_loader):
            out = model(x)
            loss = loss_func(out, torch.unsqueeze(y,dim=1).float()) * x.shape[0]
            z = torch.gt(out, 0.5)
            aaa =  torch.eq(z, y.unsqueeze(dim=1))
            acc += aaa.sum().item()
            num += x.shape[0]
            test_loss += loss.item()
        
        test_loss = test_loss/num
        test_losses.append(test_loss)
        current_acc = acc/num
        test_acc.append(current_acc)
        
        # 在评估部分添加早停逻辑
        if test_loss < min_loss:
            min_loss = test_loss
            counter = 0
            torch.save(model, './best_cats_model.pt')
            print(f"Saved new best model with test loss: {test_loss:.4f}")
        else:
            counter += 1
            
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        print(f"\rEpoch [{epoch+1}/{epochs}] Train Loss: {lss:.4f} Test Loss: {test_loss:.4f} Test Acc: {current_acc:.4f}")

    return train_losses, test_losses, test_acc


if __name__ == '__main__':
    torch.manual_seed(123)
    model = logistic_regression_model(in_num=64 * 64 * 3)
    train_data, test_data, classes = load_data_from_folder()
    
    print("Dataset Information:")
    print(f"Number of training samples: {len(train_data)}")
    print(f"Number of test samples: {len(test_data)}")
    print(f"Classes: {classes}")
    print("-" * 50)
    
    batch_size = 32
    epochs = 100
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=50)
    
    print("Training Configuration:")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {epochs}")
    print(f"Learning rate: {0.0001}")
    print(f"Weight decay: {10.0}")
    print("-" * 50)
    
    loss_func = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), 
                           lr=0.00005,  # 降低学习率
                           weight_decay=1.0)  # 降低权重衰减
    
    print("Model Architecture:")
    print(model)
    print("-" * 50)
    
    train_loss, test_loss, test_acc = train(model, train_loader, test_loader, 
                                          loss_func=loss_func,
                                          optimizer=optimizer, 
                                          epochs=epochs)
    
    draw_metrics(train_loss, test_loss, test_acc)