import numpy as np
import itertools
from scipy.special import logsumexp
from conlleval_ import evaluate
from helper import dictReverse

def forward(log_phi):
    """
    forward_matrix of shape (T, N+2), forward_matrix[-1, -1] is log_Z
    """
    N, T = log_phi.shape[0]-1, log_phi.shape[1]
    forward_matrix = np.ones((T, N+2))*np.NINF
    forward_matrix[0][0] = 0
    for i in range(1, N+2):
        forward_matrix[:, i] = logsumexp(np.expand_dims(forward_matrix[:, i-1], axis=1)+log_phi[i-1], axis=0)
        if i < N+1:
            forward_matrix[[0, -1], i] = np.NINF
    return forward_matrix


def backward(log_phi):
    """
    backward_matrix of shape (T, N+2), backward_matrix[0, 0] is log_Z
    """
    N, T = log_phi.shape[0]-1, log_phi.shape[1]
    backward_matrix = np.ones((T, N+2))*np.NINF
    backward_matrix[-1][-1] = 0
    for i in range(N, -1, -1):
        backward_matrix[:, i] = logsumexp(log_phi[i]+np.expand_dims(backward_matrix[:, i+1], axis=0), axis=1)
        if i > 0:
            backward_matrix[[0, -1], i] = np.NINF
    return backward_matrix

def loss_gradient(theta, feature2index, feature_activation, X, Y, tag2index, param=0.1):
    """
    Y in integer
    """
    log_likelihood = 0
    gradient = np.zeros(len(theta))
    exp_features = np.zeros(len(theta))
    emp_features = np.zeros(len(theta))
    T = len(tag2index)
    counter = 1

    for x, y in zip(X, Y):
        N = len(x)

        activation_map = feature_activation(x, tag2index, feature2index)
        log_phi = np.dot(activation_map, theta)

        alpha = forward(log_phi)
        beta = backward(log_phi)

        log_Z = beta[0][0]

        log_likelihood += log_phi[0, 0, y[0]]
        emp_features += activation_map[0, 0, y[0]]

        for y2 in range(1, T-1): 
            exp_features += np.exp(alpha[0, 0] + log_phi[0, 0, y2] + beta[y2, 1] - log_Z) *\
                            activation_map[0, 0, y2]

        for i in range(1, N):
            emp_features += activation_map[i, y[i-1], y[i]]
            log_likelihood += log_phi[i, y[i-1], y[i]]

            for y1, y2 in itertools.product(range(1, T-1), range(1, T-1)):
                exp_features += np.exp(alpha[y1, i] + log_phi[i, y1, y2] + beta[y2, i+1] - log_Z) *\
                                activation_map[i, y1, y2]

        for y1 in range(1, T-1):
            exp_features += np.exp(alpha[-1, N] + log_phi[N, y1, -1] + beta[-1, N+1] - log_Z) *\
                            activation_map[N, y1, -1]

        emp_features += activation_map[N, y[-1], -1]

        log_likelihood += log_phi[N, y[-1], -1]

        log_likelihood -= log_Z
        
        if counter % 5 == 0:
            print('done with the {}th instance'.format(counter))
        counter += 1
        
    gradient = exp_features - emp_features
    
    loss = -log_likelihood + param*(np.sum(theta**2))
    
    gradient += param*2*theta
    
    return loss, gradient


def viterbi(X, tag2index, feature2index, feature_activation, theta):
    """
    return Y in string
    """
    Y = list()
    # index2tag = {v-1: k for k, v in tag2index.items()}
    index2tag = dict()
    for k, v in tag2index.items():
        index2tag[v-1] = k


    for x in X:
        activation_map = feature_activation(x, tag2index, feature2index)
        log_phi = np.dot(activation_map, theta)
        N, T = len(x), len(tag2index)
        score = np.zeros((T-2, N), dtype=np.float)
        path = np.zeros((T-2, N), dtype=np.int)
        
        score[:, 0] = log_phi[0, 0, 1:-1]
        for i in range(1, N):
            cross = score[:, i-1][:, None] + log_phi[i, 1:-1, 1:-1]
            score[:, i] = np.max(cross, axis=0)
            path[:, i] = np.argmax(cross, axis=0)

        y = list()
        last_tag = np.argmax(score[:, -1] + log_phi[-1, 1:-1, -1])
        ybest = last_tag

        for i in range(len(x)-1,-1,-1):
            y.insert(0, ybest)
            ybest = path[ybest, i]

        Y.append([index2tag[idx] for idx in y])

    return Y

def eval(X, Y, tag2index, feature2index, feature_activation, theta):
    """
    Y in string
    """
    # index2tag = {value: key for key, value in tag2index.items()}
    index2tag = dictReverse(tag2index)
    Y_pred = viterbi(X, tag2index, feature2index, feature_activation, theta)
    def flatten(L):
        return [e for l in L for e in l]
    Y_pred, Y = flatten(Y_pred), flatten(Y)
    assert len(Y) == len(Y_pred)
    
    return evaluate(Y, Y_pred)

def viterbi_output(dev_out_path, X, X_str, tag2index, feature2index, feature_activation, theta):
    
    Y_pred = viterbi(X, tag2index, feature2index, feature_activation, theta)
    
    output_string = ''
    if len(X[0][0]) == 1:
        for i in range(len(X)):
            for j in range(len(X[i])):
                output_string += X_str[i][j] + ' ' + Y_pred[i][j] + '\n'
            output_string += '\n'
    elif len(X[0][0]) == 2:
        for i in range(len(X)):
            for j in range(len(X[i])):
                output_string += X_str[i][j][0] + ' ' + Y_pred[i][j] + '\n'
            output_string += '\n'
    
    with open(dev_out_path, 'w') as f:
        f.write(output_string)
    
    print('Done with writing predictions')
    return None