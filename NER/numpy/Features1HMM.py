import numpy as np
from pathlib import Path
from scipy.optimize import fmin_l_bfgs_b
import itertools
from conlleval_ import evaluate
from scipy.special import logsumexp
import time
from helper import HiddenPrints, time_elapsed

def labelled(path):
    with open(path) as f:  
        X, Y, x, y = list(), list(), list(), list()
        for line in f:
            if line == '\n':
                X.append(x)
                Y.append(y)
                x, y = list(), list()
            else:
                word, tag = line.strip().split()
                x.append(word)
                y.append(tag)
    return X, Y

def unlabelled(path):
    with open(path) as f:  
        X, x = list(), list()
        for line in f:
            if line == '\n':
                X.append(x)
                x = list()
            else:
                word = line.strip()
                x.append(word)
    return X

def read_data(root):
    train, devin, devout = root/'train', root/'dev.in', root/'dev.out'     
    return labelled(train), unlabelled(devin), labelled(devout)

def tokenize(sentence, word2index):
    return [word2index[word] if word in word2index else -1 for word in sentence]
def tag2idx(tags, tag2index):
    return [tag2index[tag] for tag in tags]

def idx_xy(X, Y, word2index=None, tag2index=None):
    if not word2index:
        vocabulary = list(set([word for sentence in X for word in sentence]))
        word2index = {word: i for i, word in enumerate(vocabulary)}
    if not tag2index:
        tags = list(set([tag for tags in Y for tag in tags]))
        tag2index = {tag: i for i, tag in enumerate(tags)}
    
    index2tag = {v:k for (k, v) in tag2index.items()}
    
    X_idx = [tokenize(sentence, word2index) for sentence in X]
    Y_idx = [tag2idx(tags, tag2index) for tags in Y]
    
    return X_idx, X, Y_idx, Y, word2index, tag2index, index2tag

def get_xy(path):
    train_ds, devin_ds, devout_ds = read_data(path)
    train_X, train_X_str, train_Y, train_Y_str, word2index, tag2index, index2tag = idx_xy(train_ds[0], train_ds[1])
    test_X, test_X_str, test_Y, test_Y_str, _, _, _ = idx_xy(devout_ds[0], devout_ds[1], word2index, tag2index)
    return train_X, train_X_str, train_Y, train_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, index2tag

def get_transition_weight(Y, tag2index):
    
    T = len(tag2index)
    count_table = np.zeros((T+1, T+1))
    for y in Y:
        count_table[-1, y[0]] += 1
        for i in range(len(y)-1):
            count_table[y[i], y[i+1]] += 1
        count_table[Y[-1], -1] += 1
            
    count_table /= count_table.sum(1)[:, None]
    
    transition_weight = np.ma.log(count_table).filled(-np.inf)
    
    return transition_weight

def get_emission_weight(X, Y, word2index, tag2index):
    
    T, V = len(tag2index), len(word2index)
    count_table = np.zeros((T, V))
    for x, y in zip(X, Y):
        for word, tag in zip(x, y):
            count_table[tag, word] += 1
            
    count_table /= count_table.sum(1)[:, None]

    emission_weight = np.ma.log(count_table).filled(-np.inf)
    
    return emission_weight

def link_weight_sum(x, transition_weight, emission_weight):

    T = transition_weight.shape[0] - 1 
    emission = np.zeros((1, T))

    if x != -1:
        emission = np.expand_dims(emission_weight[:, x], axis=0)
    transition = transition_weight[:-1, :-1]
    return transition + emission


def viterbi(X, tag2index, emission_weight, transition_weight, link_weight_sum):
    
    Y = list()
    index2tag = {value: key for key, value in tag2index.items()}

    for x in X:
        score_matrix = np.zeros((len(tag2index), len(x)))
        path_matrix = np.zeros((len(tag2index), len(x)), dtype='int')
        
        score_matrix[:, 0] = transition_weight[-1, :-1] + emission_weight[:, x[0]] if x[0] != -1 else transition_weight[-1, :-1]
        for i in range(1, len(x)):
            competitors = score_matrix[:, i-1][:, None] + link_weight_sum(x[i], transition_weight, emission_weight)
            score_matrix[:, i] = np.max(competitors, axis=0)
            path_matrix[:, i] = np.argmax(competitors, axis=0)
        
        competitors = transition_weight[:-1, -1] + score_matrix[:, -1]
        last_idx = np.argmax(competitors)
        path = [last_idx]
        for m in range(len(x)-1, 0, -1):
            path.insert(0, path_matrix[path[0], m])
        
        Y.append([index2tag[idx] for idx in path])
    
    return Y

def viterbi_output(dev_out_path, X_raw, X, tag2index, emission_weight, transition_weight, link_weight_sum):
    
    tags = viterbi(X, tag2index, emission_weight, transition_weight, link_weight_sum)
    
    output_string = ''
    for i in range(len(X)):
        for j in range(len(X[i])):
            output_string += X_raw[i][j] + ' ' + tags[i][j] + '\n'
        output_string += '\n'
    
    with open(dev_out_path, 'w') as f:
        f.write(output_string)
    
    print('Done with writing predictions')
    return None

def eval(X, Y_raw, tag2index, emission_weight, transition_weight, link_weight_sum):
    index2tag = {value: key for key, value in tag2index.items()}
    def flatten(L):
        return [e for l in L for e in l]
    Y_pred = flatten(viterbi(X, tag2index, emission_weight, transition_weight, link_weight_sum))
    Y_raw  = flatten(Y_raw)
    assert len(Y_raw) == len(Y_pred)
    return evaluate(Y_raw, Y_pred)

def Loss(X, Y, tag2index, emission_weight, transition_weight, param):
    
    loss = 0
    
    for x, y in zip(X, Y):
        pair_score = 0
        emission_score = emission_weight[y[0], x[0]]
        transition_score = transition_weight[-1, y[0]]
        pair_score += (transition_score + emission_score)
        for i in range(1, len(x)):
            emission_score = emission_weight[y[i], x[i]]
            transition_score = transition_weight[y[i-1], y[i]]
            pair_score += (transition_score + emission_score)
        
        transition_score = transition_weight[y[-1], -1]
        pair_score += transition_score
        
        _, log_Z = forward(x, tag2index, emission_weight, transition_weight)

        loss += -(pair_score - log_Z)
    
    loss += LossRegularization(emission_weight, transition_weight, param)
    
    return loss

def LossRegularization(emission_weight, transition_weight, param):
    return param*(np.sum(emission_weight[emission_weight != -np.inf]**2) +\
            np.sum(transition_weight[transition_weight != -np.inf]**2))   

def main():
    path = Path('../data/partial')
    train_X, train_X_str, train_Y, train_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, index2tag = get_xy(path)

    print('************Training Set Summary*************')
    T, V = len(tag2index), len(word2index)
    print('Number of tags: {}, Number of words: {}'.format(T, V))

    print('************Train*************')
    start = time.time()
    emission_weight = get_emission_weight(train_X, train_Y, word2index, tag2index)
    transition_weight = get_transition_weight(train_Y, tag2index)
    end = time.time()
    time_elapsed(start, end)

    print('************Saving Model Outputs*************')
    path_output = path/'dev.p2.out'
    viterbi_output(path_output, test_X_str, test_X, tag2index, emission_weight, transition_weight, link_weight_sum)
    
    prec, rec, f1 = eval(train_X, train_Y_str, tag2index, emission_weight, transition_weight, link_weight_sum)
    print('precision, recall, f1 on training set: {0} {1} {2}'.format(prec, rec, f1))
    prec, rec, f1 = eval(test_X, test_Y_str, tag2index, emission_weight, transition_weight, link_weight_sum)
    print('precision, recall, f1 on test set: {0} {1} {2}'.format(prec, rec, f1))


if __name__=='__main__':
    main()