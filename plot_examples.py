import matplotlib.pyplot as plt
import numpy as np

def plot(imgs, labels, Label_predictions):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 5, 5
    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i) 
        if len(Label_predictions)<1:
            plt.title(np.array2string(labels[i]) )
        else:
            plt.title(np.array2string(labels[i]) + '(' + np.array2string(Label_predictions[i]) + ')')
        plt.axis("off")
        plt.imshow(imgs[i].squeeze(), cmap="gray")
    plt.show()