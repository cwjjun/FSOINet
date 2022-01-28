import torch


def train(train_loader, model, criterion, sensing_rate, optimizer, device):
    model.train()
    sum_loss = 0
    for inputs in train_loader:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, sys_loss = model(inputs)
        gamma1 = torch.Tensor([0.01]).to(device)
        I = torch.eye(int(sensing_rate * 1024)).to(device)
        loss = criterion(outputs, inputs) + torch.mul(criterion(sys_loss, I), gamma1)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
    return sum_loss
