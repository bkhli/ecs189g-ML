"""
Concrete MethodModule class for a specific learning MethodModule
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
import torch
from icecream import ic
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import GloVe

from local_code.base_class.method import method
from local_code.stage_4_code.Evaluate_Accuracy import Evaluate_Accuracy
from local_code.stage_4_code.Graph_Loss import TrainLoss

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print("torch running with", device)


class JokeDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Method_LSTM(method, nn.Module):

    # it defines the the MLP model architecture, e.g.,
    # how many layers, size of variables in each layer, activation function, etc.
    # the size of the input/output portal of the model architecture should be consistent with our data input and desired output
    def __init__(self, mName, mDescription, vocab, preview):
        nn.Module.__init__(self)
        method.__init__(self, mName, mDescription)

        self.data = None
        # it defines the max rounds to train the model
        self.max_epoch = 150
        # it defines the learning rate for gradient descent based optimizer for model learning
        self.learning_rate = 5e-3
        self.batch_size = 2048

        assert vocab is not None, "[BUG] vocab is None when passed to Method_MLP"
        # print("[DEBUG] vocab type:", type(vocab))
        # print("[DEBUG] vocab has get_stoi():", hasattr(vocab, 'get_stoi'))

        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.preview = preview

        self.embedding_dim = 100
        self.hidden_size = 256  # 20
        self.dense_hidden = 128

        self.num_layers = 2
        self.bidirectional = False

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.LSTM(
            self.embedding_dim, self.hidden_size, self.num_layers, batch_first=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(
            self.hidden_size * (2 if self.bidirectional else 1), self.dense_hidden
        )
        self.fc2 = nn.Linear(self.dense_hidden, self.vocab_size)

        self.to(device)

    def collate_fn(self, batch):
        x_batch, y_batch = zip(*batch)
        lengths = [len(x) for x in x_batch]

        x_batch = [torch.tensor(x, dtype=torch.long) for x in x_batch]
        x_batch = pad_sequence(
            x_batch, batch_first=True, padding_value=self.vocab["<pad>"]
        )
        y_batch = torch.tensor(y_batch, dtype=torch.long)
        lengths = torch.tensor(lengths, dtype=torch.long)
        return x_batch, y_batch, lengths

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer
    def forward(self, x, lengths) -> torch.Tensor:
        if self.preview:  # If there is a fixed preview
            x = x.to(device)
            embeddings = self.embedding(x)
            outputs, _ = self.rnn(embeddings)
            # Outputs: [batches, seq_len, hidden]
            # Hidden_out: [layers, batches, hidden]

            if self.bidirectional:
                forward_out = outputs[
                    :, -1, : self.hidden_size
                ]  # if using a fixed preview will be -1
                backward_out = outputs[:, 0, self.hidden_size :]
                out = torch.cat([forward_out, backward_out], dim=1)
            else:
                out = outputs[:, -1, :]  # Use final output instead of hidden state

            out = self.dropout(out)
            dense_layer = self.fc1(out)
            logits = self.fc2(dense_layer)
            return logits
        else:  # variable length input
            # Im not supporting bidirectional variable yet...
            x = x.to(device)
            lengths = lengths.to(device)

            embeddings = self.embedding(x)
            outputs, _ = self.rnn(embeddings)

            # Get the output from the actual last token for each sequence
            lengths = lengths - 1
            lengths = lengths.view(-1, 1, 1).expand(-1, 1, outputs.size(2))
            out = outputs.gather(dim=1, index=lengths).squeeze(1)

            out = self.dropout(out)
            out = self.fc1(out)
            logits = self.fc2(out)
            return logits

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    # I renamed this so we could have a test mode and train mode
    # Previously conflicted with pytorch naming conventions
    def train_renamed(self, X, y):
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
                self.embedding.weight.data[idx] = glove.vectors[glove.stoi[word]].to(
                    device
                )
            else:
                self.embedding.weight.data[idx] = unk_vector.to(device)
                unk_count += 1
                # print(word) # Not as interesting as you'd think...
        print(f"{unk_count} unknown words out of a {self.vocab_size} vocab")

        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        # check here for the nn.CrossEntropyLoss doc: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        loss_function = nn.CrossEntropyLoss()  # Binary Cross Entropy Loss
        # for training accuracy investigation purpose
        accuracy_evaluator = Evaluate_Accuracy("training evaluator", "")

        # it will be an iterative gradient updating process
        # we don't do mini-batch, we use the whole input as one batch
        # you can try to split X and y into smaller-sized batches by yourself

        # train_dataset = CIFAR_Dataset(X, y)
        if self.preview:
            X_tensor = torch.tensor(np.array(X), dtype=torch.long).to(device=device)
            y_tensor = torch.tensor(np.array(y), dtype=torch.long).to(device=device)
            train_dataset = TensorDataset(X_tensor, y_tensor)
        else:
            train_dataset = JokeDataset(X, y)
        if self.preview:
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
            )  # Could replace with num_workers=0,1,2,..
        else:  # variable len
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=self.collate_fn,
            )  # Could replace with num_workers=0,1,2,..
        # For running on google colab, needs to be zero?

        loss_tracker = TrainLoss()
        for epoch in range(
            self.max_epoch
        ):  # you can do an early stop if self.max_epoch is too much...
            print(epoch)
            self.train(True)

            for idx, batch in enumerate(train_loader):
                if self.preview:
                    X, y_true = batch
                    X = X.to(device)
                    y_true = y_true.to(device)
                    y_pred = self.forward(X, None)
                else:
                    X, y_true, lengths = batch
                    X = X.to(device)
                    y_true = y_true.to(device)
                    lengths = lengths.to(device)
                    y_pred = self.forward(X, lengths)

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
                if epoch % 1 == 0 and idx == 0:
                    batch_preds = torch.argmax(y_pred, dim=1)
                    accuracy_evaluator.data = {
                        "true_y": y_true.detach().cpu().numpy(),
                        "pred_y": batch_preds.detach().cpu().numpy(),
                    }
                    # accuracy_evaluator.data = {'true_y': y_true.cpu(), 'pred_y': batch_preds.cpu()}
                    print(
                        "\nEpoch:",
                        epoch,
                        "Accuracy in batch of size",
                        self.batch_size,
                        ":",
                        accuracy_evaluator.evaluate(),
                        "Loss:",
                        train_loss.item(),
                    )
                if epoch % 10 == 0 and idx == 0:
                    self.test()
                    self.train(True)
        loss_tracker.show_graph_loss()

    def test(self):
        test_set = [
            "what did one",
            "horse walks into",
            "why did the",
            "what did the",
            "who is the",
            "why couldn't the",
        ]

        test_data = []
        for seq_str in test_set:
            tokens = seq_str.split()
            token_ids = [self.vocab[token] for token in tokens]
            test_data.append(token_ids)

        # special tokens:
        eos_id = self.vocab["<eos>"]
        pad_id = self.vocab["<pad>"]
        unk_id = self.vocab["<unk>"]
        y_preds = []

        self.eval()
        with torch.no_grad():
            for _, setup in enumerate(test_data):
                current_setup = setup.copy()
                generation = []

                for _ in range(50):
                    # The below is limiting the context, but i just realized... why would i limit context in practice? am i stupid?
                    # context = (
                    #     current_setup[-self.preview:] if len(current_setup) >= self.preview else current_setup
                    # )
                    input_tensor = torch.tensor([current_setup], dtype=torch.long).to(
                        device
                    )
                    if self.preview:
                        outputs = self.forward(input_tensor, None)
                    else:
                        length_tensor = torch.tensor(
                            [len(current_setup)], dtype=torch.long
                        ).to(device)
                        outputs = self.forward(input_tensor, length_tensor)

                    next_token = int(torch.argmax(outputs[0]).item())

                    generation.append(next_token)
                    current_setup.append(next_token)
                    # if len(current_setup) > self.preview:
                    #     current_setup = current_setup[-self.preview:]
                    if next_token == eos_id:
                        break

                generated_words = []
                for (
                    token_id
                ) in (
                    generation
                ):  # The model just genuinely likes outputting the '' char? Better tokenizing?
                    if token_id == eos_id:
                        generated_words.append("<eos>")
                    elif token_id == pad_id:
                        generated_words.append("<pad>")
                    elif token_id == unk_id:
                        generated_words.append("<unk>")
                    else:
                        # print(
                        #     f"Token ID: {token_id}, String: '{self.vocab.get_itos()[token_id]}'"
                        # )
                        generated_words.append(self.vocab.get_itos()[token_id])
                # generated_words = [
                #     self.vocab.get_itos()[token_id] for token_id in generation
                # ]
                y_preds.append(generated_words)

        out_formatted = []
        for setup, generated in zip(test_set, y_preds):
            formatted_joke = f"{setup}... {' '.join(generated)}"
            print(formatted_joke)
            out_formatted.append(formatted_joke)
        # print(self.vocab.get_itos())
        return y_preds

    def run(self):
        print("method running...")
        print("--start training...")
        self.train_renamed(self.data["X"], self.data["y"])
        print("--start testing...")
        pred_y = self.test()
        return pred_y
