import numpy as np

def sgm(x):
    return 1/(1+np.exp(-x))

def sgm_d(x):
    return sgm(x) * (1-sgm(x))

#I'm like 80% sure this function works correctly
# params are all matrix, but erros, z, bias, are n x 1
def adjust_wNb(weights_mtx, z, errors, bias, prev_a, training_rate):
    for row in range(len(weights_mtx) - 1):
        dc_da = errors[row]
        da_dz = sgm_d(z[row])
        for col in range(len(weights_mtx[0]) - 1):
            dz_dw = prev_a[col]
            
            dc_dw = dc_da * da_dz * dz_dw
            dc_db = dc_da * da_dz * 1
            
            #this is right im pretty sure
            weights_mtx[row][col] -= training_rate * dc_dw
        
        bias[row] -= training_rate * dc_db
        
    return weights_mtx, bias
            
            
            
    
    

class neural_network:
    
    # make this dynamic and able to have many layers later
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        
        self.weights_ih = np.random.rand(hidden_nodes, input_nodes) * 2 - 1
        self.weights_ho = np.random.rand(output_nodes, hidden_nodes) * 2 - 1
        
        self.bias_h = np.random.rand(hidden_nodes, 1) * 2 - 1
        self.bias_o = np.random.rand(output_nodes, 1) * 2 - 1
            
    def train(self, input_arr, target_arr):
        #turning 1D array (1 row) into nD array (1 column)
        input = np.array(input_arr)
        input = np.expand_dims(input, axis=(len(input)-1))
        target = np.array(target_arr)
        target = np.expand_dims(target, axis=(len(target)-1))
        
        #feed fwd
        z_hidden = self.weights_ih.dot(input) + self.bias_h
        a_hidden = sgm(z_hidden)
        
        z_output = self.weights_ho.dot(a_hidden) + self.bias_o
        a_output = sgm(z_output)

        #train
        errors_o = a_output - target
        
        #this step is to calc the errors of the hidden layer, watch 10.15 Neural Nets if forgot: https://www.youtube.com/watch?v=r2-P1Fi1g60&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh&index=15
        #need to transponse because we're multiplying BACKWARDS and rows/cols need to match up
        errors_h = self.weights_ho.T.dot(errors_o)

        
        
        
nn = neural_network(2, 2, 3);

input_arr = [1, 1]
target_arr = [0, 1, 2]

nn.train(input_arr, target_arr)
