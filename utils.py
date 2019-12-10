import torch
import numpy as np
import matplotlib.pyplot as plt


def get_output_for_example(model, x):
    logits = model(x.unsqueeze(0))
    probs = torch.softmax(logits, dim=-1)
    out = torch.argmax(probs, -1)

    return out


def get_accuracy(y_hat, y):
    probs = torch.softmax(y_hat, dim=-1)
    out = torch.argmax(probs, -1)
    matches = y == out
    acc = torch.sum(matches).float() / matches.numel()
    return acc


def train(model, train_dl, valid_dl, loss_fn, optimizer, num_epochs=10, print_every=100):

    train_loss = []
    valid_loss = []
    valid_acc = []

    for i in range(num_epochs):

        train_epoch_loss = []
        valid_epoch_loss = []
        valid_epoch_acc = []

        for train_step, (x, y) in enumerate(train_dl):

            model.zero_grad()
            output = model(x)
            loss = loss_fn(output.view((-1, output.size(-1))), y.view(-1))

            if train_step % print_every == 0:
                print("Epoch:\t", i, "\tStep:\t", train_step, "\tLoss:\t", float(loss))

            loss.backward()
            optimizer.step()

            train_epoch_loss.append(loss.detach().cpu().numpy())

        for train_step, (x, y) in enumerate(valid_dl):
            output = model(x).detach()
            loss = loss_fn(output.view((-1, output.size(-1))), y.view(-1))
            accuracy = get_accuracy(output, y)

            valid_epoch_loss.append(loss.detach().cpu().numpy())
            valid_epoch_acc.append(accuracy.detach().cpu().numpy())

        train_loss.append(np.mean(train_epoch_loss))
        valid_loss.append(np.mean(valid_epoch_loss))
        valid_acc.append(np.mean(valid_epoch_acc))

        print("Epoch:\t", i, "\t\t\tValid Accuracy\t", valid_acc[-1])

    return train_loss, valid_loss, valid_acc


def plot_losses(trn_loss, val_loss, val_acc):
    plt.plot(trn_loss)
    plt.plot(val_loss)
    plt.plot(val_acc)

    plt.legend(("train loss", "val loss", "val accuracy"))
    plt.xlabel('Epoch', fontsize=12)



if __name__ == '__main__':
    y = torch.Tensor([[1, 0, 1, 1, 1, 1], [1, 0, 1, 0, 1, 0]]).long()
    logits = torch.Tensor([[[1, 5], [5, 1], [1, 5], [1, 5], [1, 5], [1, 5]], [[1, 5], [5, 1], [1, 5], [5, 1], [1, 5], [5, 1]]])

    y = y.view(-1)
    logits = logits.view((-1, logits.size(-1)))

    # With PyTorch
    loss_fn = torch.nn.CrossEntropyLoss()
    print("Loss with CrossEntryopLoss()", loss_fn(logits, y))
    print("Loss with F.cross_entropy()", torch.nn.functional.cross_entropy(logits, y))




