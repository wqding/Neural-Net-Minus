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
    # c_param can either be a list of numbers representing the layers OR an instance of this class
    def __init__(self, c_param, learning_rate=None):
        
        # overwrtiting constructor if an instance of neural_network is passed
        if(isinstance(c_param, neural_network)):
            self.learning_rate = c_param.learning_rate
            self.num_layers = c_param.num_layers
            self.weights = c_param.weights
            self.biases = c_param.biases
            
        else:
            self.learning_rate = learning_rate
            self.num_layers = len(c_param)
            
            # weights is an array of m x n matrices, len = num layers - 1
            self.weights = []
            # bias is an array of 1 x n matrices, len = num layers - 1
            self.biases = []
            
            for i in range(1, self.num_layers):
                # initializing matrices representing weights between each layer with random values
                weight = np.random.rand(c_param[i], c_param[i-1]) * 2 - 1
                self.weights.append(weight)
            
            for i in range(1, self.num_layers):
                # initializing matrices representing biases between each layer with random values
                b = np.random.rand(c_param[i], 1) * 2 - 1
                self.biases.append(b)
    
    def feed_fwd(self, input_arr):
        # activations array does not include the inputs, therefore is 1 less than num layers
        layers_a = []
        layers_z = []
        #turning 1D array (1 row) into nD array (1 column)
        input = np.reshape(input_arr, (len(input_arr), 1))
                
        # minus 1 because activation layers does not include input layer
        for i in range(self.num_layers - 1):
            if i == 0:
                curr_layer_z = self.weights[i].dot(input) + self.biases[i]
                curr_layer_a = sgm(curr_layer_z)
            else:
                curr_layer_z = self.weights[i].dot(layers_a[i-1]) + self.biases[i]
                curr_layer_a = sgm(curr_layer_z)
                
            layers_z.append(curr_layer_z)
            layers_a.append(curr_layer_a)
        
        # return output layer
        return layers_a[-1]
            
    
    def train(self, input_arr, target_arr):
        #turning 1D array (1 row) into nD array (1 column)
        input = np.reshape(input_arr, (len(input_arr), 1))
        target = np.reshape(target_arr, (len(target_arr), 1))
        
        # resetting each layers activations and errors
        # activations array does not include the inputs, therefore is 1 less than num layers
        layers_a = []
        layers_z = []
        # errors array is reversed
        errors = []
        
        #feed fwd
        for i in range(self.num_layers - 1):
            if i == 0:
                curr_layer_z = self.weights[i].dot(input) + self.biases[i]
                curr_layer_a = sgm(curr_layer_z)
            else:
                curr_layer_z = self.weights[i].dot(layers_a[i-1]) + self.biases[i]
                curr_layer_a = sgm(curr_layer_z)
            
            layers_z.append(curr_layer_z)
            layers_a.append(curr_layer_a)
                

        #train
        # from num layer-1 to 0
        for i in reversed(range(self.num_layers - 1)):
            if i == self.num_layers - 2:
                curr_layer_err = target - layers_a[i]
            else:
                #mutliply by the next layers error, i-1 bc errors array is reversed                
                curr_layer_err = self.weights[i+1].T.dot(errors[(self.num_layers-3) - i])
            
            errors.append(curr_layer_err)
            curr_layer_grad = sgm_d(layers_z[i]) * curr_layer_err * self.learning_rate
            
            if i == 0:
                curr_layer_d_weight = curr_layer_grad.dot(input.T)
            else:
                curr_layer_d_weight = curr_layer_grad.dot(layers_a[i-1].T)
            
            self.biases[i] += curr_layer_grad
            self.weights[i] += curr_layer_d_weight
            
    # returns a new instance of neural network
    def copy(self):
        return neural_network(self)
        
    # # mututes the values in the weights and bias
    # def mutate(self, NN):
    #     #need to randomize some values in w and b
        


