import numpy as np

NN_ARCHITECTURE = [
    {'input_dim': 2, 'output_dim': 25, 'activation': 'relu'},
    {'input_dim': 25, 'output_dim': 50, 'activation': 'relu'},
    {'input_dim': 50, 'output_dim': 50, 'activation': 'relu'},
    {'input_dim': 50, 'output_dim': 25, 'activation': 'relu'},
    {'input_dim': 25, 'output_dim': 1, 'activation': 'sigmoid'},
]

def init_layers(nn_architecture, seed = 99):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']
        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size)*0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1)*0.1
    
    return params_values

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA*sig*(1-sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ

def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation='relu'):
    Z_curr = np.dot(W_curr, A_prev) + b_curr
    if activation is 'relu':
        activation_func = relu
    elif activation is 'sigmoid':
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
    
    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X
    
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr
        activ_function_curr = layer['activation']
        W_curr = params_values['W' + str(layer_idx)]
        b_curr = params_values['b' + str(layer_idx)]
        A_curr, Z_curr = single_layer_forward_propagation(A_prev, 
                                W_curr, b_curr, activ_function_curr)
        memory['A' + str(idx)] = A_prev
        memory['Z' + str(layer_idx)] = Z_curr
    
    return A_curr, memory

def get_cost_value(Y_hat, Y):
    m = Y_hat.shape[1]
    cost = -1/m*(np.dot(Y, np.log(Y_hat).T) + np.dot(1-Y, np.log(1-Y_hat).T))
    return np.squeeze(cost)

def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_==Y).all(axis=0).mean()

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, 
                                      A_prev, activation='relu'):
    m = A_prev.shape[1]
    
    if activation is 'relu':
        backward_activation_func = relu_backward
    elif activation is 'sigmoid':
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dz_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dz_curr, A_prev.T) / m
    db_curr = np.sum(dz_curr, axis=1, keepdims=True)/m
    dA_prev = np.dot(W_curr.T, dz_curr)

    return dA_prev, dW_curr, db_curr

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = -(np.divide(Y, Y_hat) - np.divide(1-Y, Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer['activation']

        dA_curr = dA_prev

        A_prev = memory['A' + str(layer_idx_prev)]
        Z_curr = memory['Z' + str(layer_idx_curr)]
        W_curr = params_values['W' + str(layer_idx_curr)]
        b_curr = params_values['b' + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr
        )

        grads_values['dW' + str(layer_idx_curr)] = dW_curr
        grads_values['db' + str(layer_idx_curr)] = db_curr

    return grads_values



















