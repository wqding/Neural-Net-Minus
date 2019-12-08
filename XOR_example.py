# This file demonstrates how to use the library and apply it to the XOR problem

# import library and numpy
import Neural_Net_Minus as NNM
import numpy as np

# create a neural network with 4 layers, 2 nodes in each layer except the last layer
# learning rate = 0.2
neural_network = NNM.neural_network([2, 2, 2, 1], 0.2)

# creating paired dataset
input_arr = [[1, 1], [0, 0], [1, 0], [0, 1]]
target_arr = [[0], [0], [1], [1]]

# training
for i in range(50000):
    idx = np.random.randint(0,4)
    neural_network.train(input_arr[idx], target_arr[idx])
    
# testing the neural network and printing the results
print(neural_network.feed_fwd([1, 1]))
print(neural_network.feed_fwd([1, 0]))
print(neural_network.feed_fwd([0, 1]))
print(neural_network.feed_fwd([0, 0]))

