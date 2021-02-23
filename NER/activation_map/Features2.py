import numpy as np
from pathlib import Path
from scipy.optimize import fmin_l_bfgs_b
import itertools
from core import forward, backward, loss_gradient, viterbi, eval, viterbi_output
from helper import list2dict, dictReverse


def str_emission(x, y):
    return 'emission:' + str(y) + '+' + str(x)

def str_transition(y1, y2):
    return 'transition:' + str(y1) + '+' + str(y2)

def str_emission_pos(x, y):
    return 'pos_emission:' + str(y) + '+' + str(x)



def feature2idx(words, tags, postags):
    features = list()
    T = len(tags)
    V = len(words)
    POS = len(postags)
    for y1, y2 in itertools.product(range(T), range(T)):
        features.append(str_transition(y1, y2))
    for x, y in itertools.product(range(V), range(T)):
        features.append(str_emission(x, y))
    for x, y in itertools.product(range(POS), range(T)):
        features.append(str_emission_pos(x, y))
    
    # return {f:i for i, f in enumerate(features)}
    return list2dict(features)

def feature_activation(x, tag2index, feature2index):
    N, T, F = len(x), len(tag2index), len(feature2index)
    feature_activation_map = np.zeros((N+1, T, T, F), dtype=int)
    for i, y1, y2 in itertools.product(range(N+1), range(T), range(T)):
        feature_activation_map[i, y1, y2][feature2index[str_transition(y1, y2)]] = 1
        if i < N and x[i][0] != -1:
            feature_activation_map[i, y1, y2][feature2index[str_emission(x[i][0], y2)]] = 1
            feature_activation_map[i, y1, y2][feature2index[str_emission_pos(x[i][1], y2)]] = 1
            
    return feature_activation_map

def labelled(path):
    with open(path) as f:  
        X, Y, x, y = list(), list(), list(), list()
        for line in f:
            if line == '\n':
                X.append(x)
                Y.append(y)
                x, y = list(), list()
            else:
                word, postag, tag = line.strip().split()
                x.append([word, postag])
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
                word, postag = line.strip().split()
                x.append([word, postag])
    return X

def read_data(root):
    train, devin, devout = root/'train', root/'dev.in', root/'dev.out'     
    return labelled(train), unlabelled(devin), labelled(devout)

def tokenize(sentence, word2index, postag2index):
    return [[word2index[pair[0]], postag2index[pair[1]]] if pair[0] in word2index else [-1, postag2index[pair[1]]] for pair in sentence]

def tag2idx(tags, tag2index):
    return [tag2index[tag] for tag in tags]

def idx_xy(X, Y, word2index=None, tag2index=None, postag2index=None):
    if not word2index:
        vocabulary = list(set([pair[0] for x in X for pair in x]))
        # word2index = {word: i for i, word in enumerate(vocabulary)}
        word2index = list2dict(vocabulary)
    if not tag2index:
        tags = list(set([tag for tags in Y for tag in tags]))
        tags = ['START'] + tags + ['STOP']
        # tag2index = {tag: i for i, tag in enumerate(tags)}
        tag2index = list2dict(tags)
    if not postag2index:
        postags = list(set([pair[1] for x in X for pair in x]))
        # postag2index = {postag: i for i, postag in enumerate(postags)}
        postag2index = list2dict(postags)
    
    # index2tag = {v:k for (k, v) in tag2index.items()}
    index2tag = dictReverse(tag2index)
    
    X_idx = [tokenize(sentence, word2index, postag2index) for sentence in X]
    Y_idx = [tag2idx(tags, tag2index) for tags in Y]
    
    return X_idx, X, Y_idx, Y, word2index, tag2index, postag2index, index2tag

def get_xy(path):
    train_ds, devin_ds, devout_ds = read_data(path)
    train_X, train_X_str, train_Y, train_Y_str, word2index, tag2index, postag2index, index2tag = idx_xy(train_ds[0], train_ds[1])
    test_X, test_X_str, test_Y, test_Y_str, _, _, _, _ = idx_xy(devout_ds[0], devout_ds[1], word2index, tag2index, postag2index)
    return train_X, train_X_str, train_Y, train_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, postag2index, index2tag

def test_gradient(theta, epsilon, loss_gradient, feature2index, feature_activation, X, Y, tag2index, idx=0, param=0.1):
    loss, grads = loss_gradient(theta, feature2index, feature_activation, X, Y, tag2index, param)
    theta[idx] += epsilon
    new_loss, _ = loss_gradient(theta, feature2index, feature_activation, X, Y, tag2index, param)
    print('Approximated gradient: {}, Calculated gradient: {}'.format((new_loss - loss)/epsilon, grads[idx]))


def main():
    path = Path('../data/full')
    train_X, train_X_str, train_Y, train_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, postag2index, index2tag = get_xy(path)
    feature2index = feature2idx(word2index, tag2index, postag2index)

    np.random.seed(1)
    theta = np.random.rand(len(feature2index))

    print('************Test*************')
    epsilon, idx = 0.00000001, -2
    test_gradient(theta, epsilon, loss_gradient, feature2index, feature_activation, train_X, train_Y, tag2index, idx, param=0.1)
    
    def callbackF(w):
        loss = get_loss_grad(w)[0]
        print('Loss:{0:.4f}'.format(loss))
    
    def get_loss_grad(w):
      loss, grads = loss_gradient(w, feature2index, feature_activation, train_X, train_Y, tag2index)
      return loss, grads

    print('************Train*************')
    init_w = np.zeros(len(feature2index))
    optimal_weight, final_loss, result_dict = fmin_l_bfgs_b(get_loss_grad, init_w, pgtol=0.01, callback=callbackF)
    
    path_weight = path/'best_weight_features2.npy'
    np.save(path_weight, optimal_weight)

    # optimal_weight = np.random.rand(len(feature2index))
    path_output = path/'dev.p5.CRF.f3.out'
    viterbi_output(path_output, test_X, test_X_str, tag2index, feature2index, feature_activation, optimal_weight)
    
    prec, rec, f1 = eval(train_X, train_Y_str, tag2index, feature2index, feature_activation, optimal_weight)
    print('precision, recall, f1 on training set: {0} {1} {2}'.format(prec, rec, f1))
    prec, rec, f1 = eval(test_X, test_Y_str, tag2index, feature2index, feature_activation, optimal_weight)
    print('precision, recall, f1 on test set: {0} {1} {2}'.format(prec, rec, f1))


if __name__=='__main__':
    main()