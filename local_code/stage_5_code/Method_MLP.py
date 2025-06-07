'''
Concrete MethodModule class for a specific learning MethodModule

'''

# Right now, mostly copied from stage 3 and source code provided for stage 5
# refer back to stage 3 for how to modify stuff

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_5_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_5_code.Graph_Loss import TrainLoss
import torch
from torch import nn
import numpy as np
from icecream import ic

import torch.nn.functional as F
from local_code.stage_5_code.Layers import GraphConvolutionLayer

import numpy as np
from matplotlib import pyplot as plt

import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


# device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
# print("torch running with", device)


# only working for cpu right now
torch.set_default_device("cpu")




class GCN(nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 200
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-2
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolutionLayer(nfeat, nhid)
        self.gc2 = GraphConvolutionLayer(nhid, nclass)
        self.dropout = dropout

        self.method_name = "GCN"

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
    
    def train(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0.0005)
        loss_function = F.nll_loss

        # Note - try this later
        # loss_function = nn.CrossEntropyLoss

        train_acc_lst = []
        test_acc_lst = []

        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')
        test_accuracy_evaluator = Evaluate_Accuracy('test evaluator', '')


        loss_tracker = TrainLoss()

        for epoch in range(self.max_epoch):
            idx_train = self.data['train_test']['idx_train']

            # this is like a layer:
            output = self.forward(self.data['graph']['X'], self.data['graph']['utility']['A'])

            y_pred = output[idx_train]
            y_true =  self.data['graph']['y'][idx_train]
            train_loss = loss_function(y_pred, y_true)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if epoch%1 == 0:
                loss_tracker.add_epoch(epoch, train_loss.item())
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                test_accuracy_evaluator.data = {'true_y': self.data['graph']['y'][self.data['train_test']['idx_test']].cpu(), 'pred_y': self.test()}
                train_acc = accuracy_evaluator.evaluate()
                test_acc = test_accuracy_evaluator.evaluate()
                
                train_acc_lst.append(train_acc)
                test_acc_lst.append(test_acc)

            if epoch%100 == 0:
                accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                # f1_evaluator_none.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                # f1_evaluator_macro.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                # f1_evaluator_micro.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                # f1_evaluator_weighted.data = {'true_y': y_true.cpu(), 'pred_y': y_pred.max(1)[1].cpu()}
                # print('Epoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Mutlilabel Classification:', f1_evaluator_none.evaluate(),
                #       'F1 Score - Macro:', f1_evaluator_macro.evaluate(), 'F1 Score - Micro:', f1_evaluator_micro.evaluate(),
                #       'F1 Score - Weighted:', f1_evaluator_weighted.evaluate(), 'Loss:', train_loss.item())
                print(
                    f"\nEpoch: {epoch} | Train Accuracy: {accuracy_evaluator.evaluate():.4f} | Loss: {train_loss.item():.4f}\n"
                    f"F1 Scores:\n"
                    # f"\tIndividual: {[f'{score:.4f}' for score in f1_evaluator_none.evaluate()]}\n"
                    # f"\tMacro:     {f1_evaluator_macro.evaluate():.4f}\n"
                    # f"\tMicro:     {f1_evaluator_micro.evaluate():.4f}\n"
                    # f"\tWeighted:  {f1_evaluator_micro.evaluate():.4f}"
                )
        loss_tracker.show_graph_loss()

        # plot the accuracy at the end of trainning
        plt.plot(train_acc_lst, label="train")
        plt.plot(test_acc_lst, label="test")

        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Testing Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()


    def test(self):

        output = self.forward(self.data['graph']['X'], self.data['graph']['utility']['A'])
        idx_test = self.data['train_test']['idx_test']
        y_pred = output[idx_test]
        return y_pred.max(1)[1].cpu()


    def run(self):
        print('method running...')
        print('--start training...')

        # just give it access to everything, since the layers need adjacency matrix
        self.train()

        print('--start testing...')
        pred_y = self.test()
        idx_test = self.data['train_test']['idx_test']
        return {'pred_y': pred_y, 'true_y': self.data['graph']['y'][idx_test]}