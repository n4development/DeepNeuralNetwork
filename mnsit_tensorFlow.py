import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

fig, axes = plt.subplots(3, 6, figsize=(12, 6),
                         subplot_kw={'xticks': [], 'yticks': []})
for i in range(0, 5):
    data = mnist.train.images[i]
    pixels = data[0:]
    print(pixels.shape)
    pixels = np.array(pixels, dtype='float32')
    pixels = pixels.reshape((28, 28))
    plt.title("Label is {label}".format(label=mnist.train.labels[i]))
    plt.imshow(pixels, cmap="gray")
    plt.show()
