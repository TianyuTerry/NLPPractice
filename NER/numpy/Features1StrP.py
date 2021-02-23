import numpy as np
from pathlib import Path
import itertools
import random
from conlleval_ import evaluate
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

def read_data(root):
    train, dev, test = root/'train', root/'dev', root/'test'     
    return labelled(train), labelled(dev), labelled(test)

def tokenize(sentence, word2index):
    return [word2index[word] if word in word2index else -1 for word in sentence]
def tag2idx(tags, tag2index):
    return [tag2index[tag] for tag in tags]
def idx2tag(tags_idx, index2tag):
    return [index2tag[tag_idx] for tag_idx in tags_idx]

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
    train_ds, dev_ds, test_ds = read_data(path)
    train_X, train_X_str, train_Y, train_Y_str, word2index, tag2index, index2tag = idx_xy(train_ds[0], train_ds[1])
    dev_X, dev_X_str, dev_Y, dev_Y_str, _, _, _ = idx_xy(dev_ds[0], dev_ds[1], word2index, tag2index)
    test_X, test_X_str, test_Y, test_Y_str, _, _, _ = idx_xy(test_ds[0], test_ds[1], word2index, tag2index)
    return train_X, train_X_str, train_Y, train_Y_str, dev_X, dev_X_str, dev_Y, dev_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, index2tag

def link_weight_sum(x, transition_weight, emission_weight):

    T = transition_weight.shape[0] - 1 
    emission = np.zeros((1, T))

    if x != -1:
        emission = np.expand_dims(emission_weight[:, x], axis=0)
    transition = transition_weight[:-1, :-1]
    
    return transition + emission

def viterbi(x, tag2index, emission_weight, transition_weight, link_weight_sum):

    score_matrix = np.zeros((len(tag2index), len(x)))
    path_matrix = np.zeros((len(tag2index), len(x)), dtype='int')
    
    score_matrix[:, 0] = transition_weight[-1, :-1] + emission_weight[:, x[0]] if x[0] != -1 else transition_weight[-1, :-1]
    for i in range(1, len(x)):
        competitors = score_matrix[:, i-1][:, None] + link_weight_sum(x[i], transition_weight, emission_weight)
        score_matrix[:, i] = np.max(competitors, axis=0)
        path_matrix[:, i] = np.argmax(competitors, axis=0)
    
    competitors = transition_weight[:-1, -1] + score_matrix[:, -1]
    last_idx = np.argmax(competitors)
    y = [last_idx]
    for m in range(len(x)-1, 0, -1):
        y.insert(0, path_matrix[y[0], m])
    
    return y

def viterbi_output(dev_out_path, X_raw, X, tag2index, emission_weight, transition_weight, link_weight_sum):

    index2tag = {value: key for key, value in tag2index.items()}
    Y = [viterbi(x, tag2index, emission_weight, transition_weight, link_weight_sum) for x in X]
    tags = [idx2tag(y, index2tag) for y in Y]
    
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
    Y_pred = flatten([idx2tag(viterbi(x, tag2index, emission_weight, transition_weight, link_weight_sum), index2tag) for x in X])
    Y_raw  = flatten(Y_raw)
    assert len(Y_raw) == len(Y_pred)
    return evaluate(Y_raw, Y_pred)

def Loss(X, Y, tag2index, word2index, transition_weight, emission_weight, link_weight_sum):
    
    loss = 0
    
    for x, y in zip(X, Y):
        transition_count, emission_count = feature_count(x, y, tag2index, word2index)
        loss -= (np.sum(transition_count*transition_weight) + np.sum(emission_count*emission_weight))
        y_pred = viterbi(x, tag2index, emission_weight, transition_weight, link_weight_sum)
        transition_count_pred, emission_count_pred = feature_count(x, y_pred, tag2index, word2index)
        loss += (np.sum(transition_count_pred*transition_weight) +np.sum(emission_count_pred*emission_weight))
    
    return loss

def feature_count(x, y, tag2index, word2index):
    
    T, V, N = len(tag2index), len(word2index), len(x)
    transition_count, emission_count = np.zeros((T+1, T+1)), np.zeros((T, V))

    transition_count[-1, y[0]] += 1
    for i in range(1, N):
        transition_count[y[i-1], y[i]] += 1
    transition_count[y[-1], -1] += 1

    for i in range(N):
        emission_count[y[i], x[i]] += 1

    return transition_count, emission_count

def trainOld(X, Y, tag2index, word2index, threshold, random_seed=1):

    T, V, D, counter = len(tag2index), len(word2index), len(X), 1
    transition_weight, emission_weight = np.zeros((T+1, T+1)), np.zeros((T, V))
    np.random.seed(random_seed)

    while True:
        k = random.randint(0, D)
        x, y = X[k], Y[k]
        transition_count, emission_count = feature_count(x, y, tag2index, word2index)
        y_pred = viterbi(x, tag2index, emission_weight, transition_weight, link_weight_sum)
        transition_count_pred, emission_count_pred = feature_count(x, y_pred, tag2index, word2index)
        if np.allclose(transition_count, transition_count_pred, rtol=0, atol=threshold) and np.allclose(emission_count, emission_count_pred, rtol=0, atol=threshold):
            print(np.sum(np.abs(transition_count_pred - transition_count)), np.sum(np.abs(emission_count_pred - emission_count)))
            break
        else:
            transition_weight += (transition_count - transition_count_pred)
            emission_weight += (emission_count - emission_count_pred)
        if counter % 500 == 0:
            loss = Loss(X, Y, tag2index, word2index, transition_weight, emission_weight, link_weight_sum)
            print('{} updates done, loss: {:.4f}'.format(counter, loss))
        counter += 1

    return transition_weight, emission_weight


def train(X, Y, X_test, Y_test_raw, tag2index, word2index, iteration=20, random_seed=1):

    T, V, D = len(tag2index), len(word2index), len(X)
    f1_opti = 0
    transition_weight, emission_weight = np.zeros((T+1, T+1)), np.zeros((T, V))
    np.random.seed(random_seed)

    for i in range(iteration):
        for j in range(D):
            k = random.randint(0, D-1)
            x, y = X[k], Y[k]
            transition_count, emission_count = feature_count(x, y, tag2index, word2index)
            y_pred = viterbi(x, tag2index, emission_weight, transition_weight, link_weight_sum)
            transition_count_pred, emission_count_pred = feature_count(x, y_pred, tag2index, word2index)
            transition_weight += (transition_count - transition_count_pred)
            emission_weight += (emission_count - emission_count_pred)
        loss = Loss(X, Y, tag2index, word2index, transition_weight, emission_weight, link_weight_sum)
        with HiddenPrints():
            prec, rec, f1 = eval(X_test, Y_test_raw, tag2index, emission_weight, transition_weight, link_weight_sum)
        print('training epoch: {} , training loss: {:.4f} test F1: {:.4f}'.format(i+1, loss, f1))
        if f1 > f1_opti:
            transition_weight_opti, emission_weight_opti, f1_opti = transition_weight, emission_weight, f1
            print('better parameters found!')

    return transition_weight_opti, emission_weight_opti

def trainDecay(X, Y, X_dev, Y_dev_raw, tag2index, word2index, iteration=20, random_seed=1):

    T, V, D = len(tag2index), len(word2index), len(X)
    f1_opti, counter = 0, 1
    transition_weight, emission_weight = np.zeros((T+1, T+1)), np.zeros((T, V))
    np.random.seed(random_seed)

    for i in range(iteration):
        for j in range(D):
            k = random.randint(0, D-1)
            x, y = X[k], Y[k]
            transition_count, emission_count = feature_count(x, y, tag2index, word2index)
            y_pred = viterbi(x, tag2index, emission_weight, transition_weight, link_weight_sum)
            transition_count_pred, emission_count_pred = feature_count(x, y_pred, tag2index, word2index)
            transition_weight += (transition_count - transition_count_pred)*1/counter
            emission_weight += (emission_count - emission_count_pred)*1/counter
        loss = Loss(X, Y, tag2index, word2index, transition_weight, emission_weight, link_weight_sum)
        with HiddenPrints():
            prec, rec, f1 = eval(X_dev, Y_dev_raw, tag2index, emission_weight, transition_weight, link_weight_sum)
        print('training epoch: {} , training loss: {:.4f} dev F1: {:.4f}'.format(i+1, loss, f1))
        counter += 1
        if f1 > f1_opti:
            transition_weight_opti, emission_weight_opti, f1_opti = transition_weight.copy(), emission_weight.copy(), f1
            print('better parameters found!')

    return transition_weight_opti, emission_weight_opti

def main():
    path = Path('../data/partial')
    train_X, train_X_str, train_Y, train_Y_str, dev_X, dev_X_str, dev_Y, dev_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, index2tag = get_xy(path)

    print('************Training Set Summary*************')
    T, V = len(tag2index), len(word2index)
    print('Number of tags: {}, Number of words: {}'.format(T, V))

    print('************Train*************')
    start = time.time()
    optimal_transition_weight, optimal_emission_weight = trainDecay(train_X, train_Y, dev_X, dev_Y_str, tag2index, word2index, iteration=20, random_seed=1)
    end = time.time()
    time_elapsed(start, end)
    
    print('************Saving Model Parameters*************')
    path_transition = path/'best_weight_features1_transition_strp.npy'
    path_emission = path/'best_weight_features1_emission_strp.npy'
    np.save(path_transition, optimal_transition_weight)
    np.save(path_emission, optimal_emission_weight)
    
    print('************Evaluation*************')
    prec, rec, f1 = eval(train_X, train_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, link_weight_sum)
    print('precision, recall, f1 on training set: {0} {1} {2}'.format(prec, rec, f1))
    prec, rec, f1 = eval(dev_X, dev_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, link_weight_sum)
    print('precision, recall, f1 on development set: {0} {1} {2}'.format(prec, rec, f1))
    prec, rec, f1 = eval(test_X, test_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, link_weight_sum)
    print('precision, recall, f1 on test set: {0} {1} {2}'.format(prec, rec, f1))


if __name__=='__main__':
    main()