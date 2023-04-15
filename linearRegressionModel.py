import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

weight = 0.7  # m
bias = 0.3  # b
# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


def plot_predictions(train_data=X_train,
                     train_labels=y_train, test_data=X_test,
                     test_labels=y_test, predictions=None):
    """
    Plots Training Data, test data and compares predictions
    :type predictions: object
    :param train_data:
    :param train_labels:
    :param test_data:
    :param test_labels:
    :param predictions:
    :return:
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c='b', s=4, label='Training Data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing Data')

    if predictions is not None:
        # Plotting predictions if they exist
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
    plt.legend(prop={'size': 14})
    plt.show()


class LinearRegressionModel(
    nn.Module):  # <- almost everything in PyTorch is a nn.Module (think of this as neural network lego blocks)
    # https://github.com/mrdbourke/pytorch-deep-learning/blob/main/01_pytorch_workflow.ipynb
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(1,  # <- start with random weights (this will get adjusted as the model learns)
                        dtype=torch.float),  # <- PyTorch loves float32 by default
            requires_grad=True)  # <- can we update this value with gradient descent?)

        self.bias = nn.Parameter(
            torch.randn(1,  # <- start with random bias (this will get adjusted as the model learns)
                        dtype=torch.float),  # <- PyTorch loves float32 by default
            requires_grad=True)  # <- can we update this value with gradient descent?))

    # Forward defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # <- "x" is the input data (e.g. training/testing features)
        return self.weights * x + self.bias  # <- this is the linear regression formula (y = m*x + b)
        #return (self.bias-self.weights)**2

torch.manual_seed(42)
model = LinearRegressionModel()

with torch.inference_mode():
    y_preds = model(X_test)

loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)
epochs = 150
epoch_count = []
loss_values = []
test_loss_values = []
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    #print(f'Loss: {loss}')
    with torch.inference_mode():
        # 1. forward pass
        test_pred = model(X_test)
        # 2. calculate the loss
        test_loss = loss_fn(test_pred,y_test)
    if epoch % 10 == 0:
        #print(f"Epoch: {epoch} | Loss: {loss} | Test Loss {test_loss}")
        epoch_count.append(epoch)
        loss_values.append(loss.clone().detach())
        test_loss_values.append(test_loss)
        #print(f"Epoch: {epoch} | Loss: {loss} | Test Loss {test_loss}")

with torch.inference_mode():
    y_preds_new = model(X_test)

plot_predictions(predictions=y_preds_new)

### Plotting loss curves:
plt.plot(epoch_count, loss_values, label='Train Loss')
plt.plot(epoch_count, test_loss_values, label='Test Loss')
plt.title('Training and test loss curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# MODEL_PATH = Path('models') # Creating a new folder
# MODEL_PATH.mkdir(parents=True,exist_ok=True)
# MODEL_NAME = '01_pytorch_workflow_model_0.pth'
# MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# # Save the model state dict
# print(f"Saving model to: {MODEL_SAVE_PATH}")
# torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)
# loaded_model = LinearRegressionModel()
# loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
# print(loaded_model.state_dict())
#
# loaded_model.eval()
# with torch.inference_mode():
#     loaded_model_preds = loaded_model(X_test)
# model.eval()
# with torch.inference_mode():
#     y_preds = model(X_test)
# print(loaded_model_preds)
# print(y_preds == loaded_model_preds) # Checking if the models match based on the prediction values
