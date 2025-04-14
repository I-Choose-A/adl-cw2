import torch

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(2025)

loss_function = torch.nn.CrossEntropyLoss()


def eval_classifier(model, x, y):
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    model.eval()

    with torch.no_grad():
        y_pred = model(x)
        loss = loss_function(y_pred, y)

        # compute acc
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == y).sum().item()

    return correct, loss.item()
