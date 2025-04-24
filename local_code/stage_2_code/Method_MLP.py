'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_2_code.Evaluate_F1 import Evaluate_F1_None, Evaluate_F1_Macro, Evaluate_F1_Micro, Evaluate_F1_Weighted
import torch
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        # check here for nn.Linear doc: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.fc_layer_1 = nn.Linear(784, 256).to(device)
        # check here for nn.ReLU doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
        self.activation_func_1 = nn.ReLU().to(device)
        self.fc_layer_2 = nn.Linear(256, 64).to(device)
        # check here for nn.Softmax doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        self.activation_func_2 = nn.ReLU().to(device)
        self.fc_layer_3 = nn.Linear(64, 10).to(device)
        self.activation_func_3 = nn.Softmax(dim=1).to(device)


    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x) -> torch.Tensor:
        '''Forward propagation'''
        # hidden layer embeddings
        h = self.activation_func_1(self.fc_layer_1(x))
        # outout layer result
        # self.fc_layer_2(h) will be a n x 10 tensor
        # n (denotes the input instance number): 0th dimension; 10 (denotes the class number): 1st dimension
        # we do softmax along dim=1 to get the normalized classification probability distributions for each instance
        # y_pred = self.activation_func_2(self.fc_layer_2(h))
        h2 = self.activation_func_2(self.fc_layer_2(h))
        y_pred = self.activation_func_3(self.fc_layer_3(h2))
        return y_pred

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        X = torch.FloatTensor(np.array(X)).to(device)
        # convert y to torch.tensor as well
        y_true = torch.LongTensor(np.array(y)).to(device)

        # check here for the torch.optim doc: https://pytorch.org/docs/stable/optim.html
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()
        #loss_function = nn.MSELoss()

        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        f1_evaluator_none = Evaluate_F1_None('multilabel classification', '')
        f1_evaluator_macro = Evaluate_F1_Macro('multilabel classification', '')
        f1_evaluator_micro = Evaluate_F1_Micro('multilabel classification', '')
        f1_evaluator_weighted = Evaluate_F1_Weighted('multilabel classification', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself
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

            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                f1_evaluator_none.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                f1_evaluator_macro.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                f1_evaluator_micro.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                f1_evaluator_weighted.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Mutlilabel Classification:', f1_evaluator_none.evaluate(), 
                      'F1 Score - Macro:', f1_evaluator_macro.evaluate(), 'F1 Score - Micro:', f1_evaluator_micro.evaluate(), 
                      'F1 Score - Weighted:', f1_evaluator_weighted.evaluate(), 'Loss:', train_loss.item())
    
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