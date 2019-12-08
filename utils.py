import torch
import numpy as np


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

    for i in range(num_epochs):

        for train_step, (x, y) in enumerate(train_dl):

            model.zero_grad()
            output = model(x)
            loss = loss_fn(output.view((-1, output.size(-1))), y.view(-1))

            if train_step % print_every == 0:
                print("Epoch:\t", i, "\tStep:\t", train_step, "\tLoss:\t", float(loss))

            loss.backward()
            optimizer.step()

        accuracies = []
        for train_step, (x, y) in enumerate(valid_dl):
            output = model(x).detach()
            accuracy = get_accuracy(output, y)
            accuracies.append(accuracy.cpu().numpy())

        print("Epoch:\t", i, "\t\t\tValid Accuracy\t", np.mean(accuracies))

