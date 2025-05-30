'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_3_code.Graph_Loss import TrainLoss
import torch
from torch import nn
import numpy as np
from icecream import ic

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("torch running with", device)

class Method_MLP(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 500
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        n1 = 5
        n2 = 10

        # input Image size: 28x28 (the images are gray-scale with only one channel)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n1, kernel_size=5).to(device)  # doc: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        # output size: 28 + 2 * 0 - 5 + 1 = 24        
        self.pool1 = nn.MaxPool2d(kernel_size=2).to(device)  # doc: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d
        # output size: 12 x 12
        self.conv2 = nn.Conv2d(in_channels=n1, out_channels=n2, kernel_size=5).to(device)
        # output size: 8 x 8
        self.pool2 = nn.MaxPool2d(kernel_size=2).to(device)
        # output size: 4 x 4

        # completely flat to 1D
        self.flatten = nn.Flatten().to(device)
        
        n3 = 4 * 4 * n2
        self.fc_layer_1 = nn.Linear(n3, 64).to(device)
        self.activation_func_1 = nn.ReLU().to(device)

        self.fc_layer_2 = nn.Linear(64, 10).to(device)


    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x) -> torch.Tensor:
        '''Forward propagation'''
        # hidden layer embeddings
        # ic(x.shape)
        
        imagefeatures1 = self.pool1(self.conv1(x))
        imagefeatures2 = self.pool2(self.conv2(imagefeatures1))
        
        flattend = self.flatten(imagefeatures2)

        h = self.activation_func_1(self.fc_layer_1(flattend))
        y_pred = self.fc_layer_2(h)
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        ic(len(y))
        ic(len(X))
        # ic(X[0])
        ic(y[0])

        # X is image
        # y is label

        X = torch.FloatTensor(np.array(X)).to(device)
        ic(X.shape)
        # convert y to torch.tensor as well
        y_true = torch.LongTensor(np.array(y)).to(device)

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
        loss_tracker = TrainLoss()
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
            y_pred = self.forward(X)
            # calculate the training loss
            train_loss = loss_function(y_pred, y_true)

            # check here for the gradient init doc: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
            optimizer.zero_grad()
            # check here for the loss.backward doc: https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
            # do the error backpropagation to calculate the gradients
            train_loss.backward()
            # check here for the opti.step doc: https://pytorch.org/docs/stable/optim.html
            # update the variables according to the optimizer and the gradients calculated by the above loss.backward function
            optimizer.step()
            if epoch%5 ==0:
                loss_tracker.add_epoch(epoch, train_loss.item())
            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
        loss_tracker.show_graph_loss()
    def test(self, X):
        # do the testing, and result the result
        y_pred = self.forward(torch.FloatTensor(np.array(X)).to(device))
        # convert the probability distributions to the corresponding labels
        # instances will get the labels corresponding to the largest probability

        return y_pred.max(1)[1].cpu()
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}