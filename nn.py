import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

class neural_network:
    
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        
        self.weights_ih = np.random.rand(hidden_nodes, input_nodes) * 2 - 1
        self.weights_ho = np.random.rand(output_nodes, hidden_nodes) * 2 - 1
        
        self.bias_h = np.random.rand(hidden_nodes, 1) * 2 - 1
        self.bias_o = np.random.rand(output_nodes, 1) * 2 - 1
        
    def feed_fwd(self, input_arr):
        input = np.array(input_arr)
        input = np.expand_dims(input, axis=(len(input)-1))
        z_hidden = self.weights_ih.dot(input) + self.bias_h
        #working
        print(z_hidden)
        a_hidden = sigmoid(z_hidden)
        print(a_hidden)
        

        
        
        
nn = neural_network(2, 2, 3);

input_arr = [1, 1]

nn.feed_fwd(input_arr)
