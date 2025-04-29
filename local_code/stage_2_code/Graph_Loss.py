from matplotlib import pyplot as plt
import numpy as np


class TrainLoss:
    def __init__(self):
        self.epochs = []
        self.loss = []

    def add_epoch(self, epoch, loss):
        self.epochs.append(epoch)
        self.loss.append(loss)

    def show_graph_loss(self):
        plt.plot(self.epochs, self.loss, label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.grid(True)
        plt.legend()
        plt.show()

    def save_data(self, filename):
        np.save(filename, [self.epochs, self.loss])

