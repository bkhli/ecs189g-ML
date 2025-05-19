'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Graph_Loss import TrainLoss
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import GloVe
import numpy as np
from icecream import ic

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("torch running with", device)

class Method_MLP(method, nn.Module):


    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, vocab):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)

        self.data = None
        # it defines the max rounds to train the model
        self.max_epoch = 500
        # it defines the learning rate for gradient descent based optimizer for model learning
        self.learning_rate = 1e-3
        self.batch_size = 64

        assert vocab is not None, "[BUG] vocab is None when passed to Method_MLP"
        # print("[DEBUG] vocab type:", type(vocab))
        # print("[DEBUG] vocab has get_stoi():", hasattr(vocab, 'get_stoi'))

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = 100
        self.hidden_size = 20

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.hidden_size, 1)


    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x) -> torch.Tensor:
        embeddings = self.embedding(x)
        outputs, hidden_out = self.rnn(embeddings) 
        # Outputs: [batches, seq_len, hidden]
        # Hidden_out: [layers, batches, hidden]

        out = hidden_out[-1]
        # out = self.dropout(out)
        logits = self.fc(out)
        return logits.squeeze(1) # Because they're nested


    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        # ic(len(y))
        # ic(len(X))
        # ic(X[0])
        # ic(y[0])

        # X is image
        # y is label

        glove = GloVe(name="6B", dim=self.embedding_dim)
        unk_vector = torch.zeros(self.embedding_dim)
        unk_count = 0
        for word, idx in self.vocab.get_stoi().items():
            if word in glove.stoi:
                self.embedding.weight.data[idx] = glove.vectors[glove.stoi[word]]
            else:
                self.embedding.weight.data[idx] = unk_vector
                unk_count += 1
                # print(word) # Not as interesting as you'd think...
        print(f"{unk_count} unknown words")

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself

        # train_dataset = CIFAR_Dataset(X, y)
        X_tensor = torch.tensor(np.array( X ), dtype=torch.long).to(device=device)
        y_tensor = torch.tensor(np.array( y ), dtype=torch.float32).to(device=device)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        loss_tracker = TrainLoss()
        for epoch in range(self.max_epoch): # you can do an early stop if self.max_epoch is too much...
            print(epoch)

            for idx, (X, y_true) in enumerate(train_loader):
                # ic(X.shape)
                # ic(y_true.shape)

                # get the output, we need to covert X into torch.tensor so pytorch algorithm can operate on it
                y_pred = self.forward(X)
                # ic(y_pred.shape)

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

                if idx == 0:
                    loss_tracker.add_epoch(epoch, train_loss.item())
                if epoch%10 == 0 and idx == 0:
                    batch_preds = torch.round(torch.sigmoid(y_pred))

                    accuracy_evaluator.data = {
                        'true_y': y_true.detach().cpu().numpy(),
                        'pred_y': batch_preds.detach().cpu().numpy()
                    }
                    # accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': batch_preds.cpu()}
                    print('\nEpoch:', epoch, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
                if epoch%50 == 0 and idx ==0:
                    test_preds = self.test(self.data['test']['X']) 
                    accuracy_evaluator.data = {
                        'true_y': self.data['test']['y'],
                        'pred_y': test_preds.detach().cpu().numpy()
                    }
                    test_acc = accuracy_evaluator.evaluate()
                    print('Test Accuracy:', test_acc)
        loss_tracker.show_graph_loss()



    def test(self, X):
        X_tensor = torch.tensor(np.array( X ), dtype=torch.long).to(device=device) 
        test_dataset = TensorDataset(X_tensor)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=0)

        y_preds = []

        # self.eval()
        with torch.no_grad():
            for (X,) in test_loader:
                outputs = self.forward(X)
                batch_preds = torch.round(torch.sigmoid(outputs))
                y_preds.append(batch_preds)
        return torch.cat(y_preds)


    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
