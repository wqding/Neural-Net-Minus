import numpy as np

def sgm(x):
    return 1/(1+np.exp(-x))

def sgm_d(x):
    return sgm(x) * (1-sgm(x))

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    # print(x)
    x[x<=0] = 0
    x[x>0] = 1
    return x


class neural_network:
    # make this dynamic and able to have many layers later
    def __init__(self, layers, learning_rate):
        self.learning_rate = learning_rate
        self.num_layers = len(layers)
        
        # errors array is reversed
        self.errors = []
        
        # activations array does not include the inputs, therefore is 1 less than num layers
        self.layers_a = []
        self.layers_z = []
        
        # weights is an array of m x n matrices, len = num layers - 1
        self.weights = []
         # bias is an array of 1 x n matrices, len = num layers - 1
        self.bias = []
        
        for i in range(1, self.num_layers):
            weight = np.random.rand(layers[i], layers[i-1]) * 2 - 1
            self.weights.append(weight)
        
        for i in range(1, self.num_layers):
            b = np.random.rand(layers[i], 1) * 2 - 1
            self.bias.append(b)
    
    def feed_fwd(self, input_arr):
        self.layers_a = []
        self.layers_z = []
        #turning 1D array (1 row) into nD array (1 column)
        input = np.reshape(input_arr, (len(input_arr), 1))
                
        # minus 1 because activation layers does not include input layer
        for i in range(self.num_layers - 1):
            if i == 0:
                curr_layer_z = self.weights[i].dot(input) + self.bias[i]
                curr_layer_a = sgm(curr_layer_z)
            else:
                curr_layer_z = self.weights[i].dot(self.layers_a[i-1]) + self.bias[i]
                curr_layer_a = sgm(curr_layer_z)
                
            self.layers_z.append(curr_layer_z)
            self.layers_a.append(curr_layer_a)
        
        # return output layer
        return self.layers_a[-1]
            
    
    def train(self, input_arr, target_arr):
        #turning 1D array (1 row) into nD array (1 column)
        input = np.reshape(input_arr, (len(input_arr), 1))
        target = np.reshape(target_arr, (len(target_arr), 1))
        
        # resetting each layers activations and errors
        self.layers_a = []
        self.layers_z = []
        self.errors = []
        
        #feed fwd
        for i in range(self.num_layers - 1):
            if i == 0:
                curr_layer_z = self.weights[i].dot(input) + self.bias[i]
                curr_layer_a = sgm(curr_layer_z)
            else:
                curr_layer_z = self.weights[i].dot(self.layers_a[i-1]) + self.bias[i]
                curr_layer_a = sgm(curr_layer_z)
            
            self.layers_z.append(curr_layer_z)
            self.layers_a.append(curr_layer_a)
                

        #train
        # from num layer-1 to 0
        for i in reversed(range(self.num_layers - 1)):
            if i == self.num_layers - 2:
                curr_layer_err = target - self.layers_a[i]
            else:
                #mutliply by the next layers error, i-1 bc errors array is reversed                
                curr_layer_err = self.weights[i+1].T.dot(self.errors[(self.num_layers-3) - i])
            
            self.errors.append(curr_layer_err)
            curr_layer_grad = sgm_d(self.layers_z[i]) * curr_layer_err * self.learning_rate
            
            if i == 0:
                curr_layer_d_weight = curr_layer_grad.dot(input.T)
            else:
                curr_layer_d_weight = curr_layer_grad.dot(self.layers_a[i-1].T)
            
            self.bias[i] += curr_layer_grad
            self.weights[i] += curr_layer_d_weight


