from matplotlib import pyplot as plt
import numpy as np

epoch_e_2, loss_e_2 = np.load("No-Decay.npy") # 96.99 acc
epoch_e_3, loss_e_3 = np.load("1-Decay.npy") # 97 acc
epoch_e_4, loss_e_4 = np.load("2-Decay.npy") # 

plt.plot(epoch_e_2, loss_e_2, label='No Decay')
plt.plot(epoch_e_3, loss_e_3, label='Weight Decay=1e-4')
plt.plot(epoch_e_4, loss_e_4, label='Weight Decay=1e-2')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale("log")
plt.title('Training Loss')
plt.grid(True)
plt.legend()
plt.show()
