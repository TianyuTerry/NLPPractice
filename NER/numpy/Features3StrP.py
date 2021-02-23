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
                word, postag, tag = line.strip().split()
                x.append([word, postag])
                y.append(tag)
    return X, Y

def read_data(root):
    train, dev, test = root/'train', root/'dev', root/'test'     
    return labelled(train), labelled(dev), labelled(test)

def tokenize(sentence, word2index, postag2index):
    idx, word, postag = list(), -1, -1
    for pair in sentence:
        if pair[0] in word2index:
            word = word2index[pair[0]]
        if pair[1] in postag2index:
            postag = postag2index[pair[1]]
        idx.append([word, postag])
    return idx

def tag2idx(tags, tag2index):
    return [tag2index[tag] for tag in tags]

def idx2tag(tags_idx, index2tag):
    return [index2tag[tag_idx] for tag_idx in tags_idx]

def idx_xy(X, Y, word2index=None, tag2index=None, postag2index=None):
    if not word2index:
        vocabulary = list(set([pair[0] for x in X for pair in x]))
        word2index = {word: i for i, word in enumerate(vocabulary)}
    if not tag2index:
        tags = list(set([tag for tags in Y for tag in tags]))
        tag2index = {tag: i for i, tag in enumerate(tags)}
    if not postag2index:
        postags = list(set([pair[1] for x in X for pair in x]))
        postag2index = {postag: i for i, postag in enumerate(postags)}
    
    index2tag = {v:k for (k, v) in tag2index.items()}
    
    X_idx = [tokenize(sentence, word2index, postag2index) for sentence in X]
    Y_idx = [tag2idx(tags, tag2index) for tags in Y]
    
    return X_idx, X, Y_idx, Y, word2index, tag2index, postag2index, index2tag

def get_xy(path):
    train_ds, dev_ds, test_ds = read_data(path)
    train_X, train_X_str, train_Y, train_Y_str, word2index, tag2index, postag2index, index2tag = idx_xy(train_ds[0], train_ds[1])
    dev_X, dev_X_str, dev_Y, dev_Y_str, _, _, _, _ = idx_xy(dev_ds[0], dev_ds[1], word2index, tag2index, postag2index)
    test_X, test_X_str, test_Y, test_Y_str, _, _, _, _ = idx_xy(test_ds[0], test_ds[1], word2index, tag2index, postag2index)
    return train_X, train_X_str, train_Y, train_Y_str, dev_X, dev_X_str, dev_Y, dev_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, postag2index, index2tag

def link_weight_sum(x, transition_weight, emission_weight, emission_weight_pos, combination_weight, combination_weight_pos):

    T = transition_weight.shape[0] - 1 
    emission = np.zeros((1, T))
    combination = np.zeros((T, T))

    if x[0] != -1:
        emission += np.expand_dims(emission_weight[:, x[0]], axis=0)
        combination += combination_weight[:-1, :, x[0]]
    if x[1] != -1:
        emission += np.expand_dims(emission_weight_pos[:, x[1]], axis=0)
        combination += combination_weight_pos[:-1, :, x[1]]
    
    transition = transition_weight[:-1, :-1]

    return transition + emission + combination

def viterbi(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum):

    score_matrix = np.zeros((len(tag2index), len(x)))
    path_matrix = np.zeros((len(tag2index), len(x)), dtype='int')
    
    if x[0][0] != -1 and x[0][1] != -1:
        score_matrix[:, 0] = transition_weight[-1, :-1] + combination_weight[-1, :, x[0][0]] + combination_weight_pos[-1, :, x[0][1]] + emission_weight[:, x[0][0]] + emission_weight_pos[:, x[0][1]]  
    elif x[0][0] == -1 and x[0][1] != -1:
        score_matrix[:, 0] = transition_weight[-1, :-1] + emission_weight_pos[:, x[0][1]] + combination_weight_pos[-1, :, x[0][1]]
    elif x[0][1] == -1 and x[0][0] != -1:
        score_matrix[:, 0] = transition_weight[-1, :-1] + combination_weight[-1, :, x[0][0]] + emission_weight[:, x[0][0]]
    else:
        score_matrix[:, 0] = transition_weight[-1, :-1]
    for i in range(1, len(x)):
        competitors = score_matrix[:, i-1][:, None] + link_weight_sum(x[i], transition_weight, emission_weight, emission_weight_pos, combination_weight, combination_weight_pos)
        score_matrix[:, i] = np.max(competitors, axis=0)
        path_matrix[:, i] = np.argmax(competitors, axis=0)
    
    competitors = transition_weight[:-1, -1] + score_matrix[:, -1]
    last_idx = np.argmax(competitors)
    y = [last_idx]
    for m in range(len(x)-1, 0, -1):
        y.insert(0, path_matrix[y[0], m])
    
    return y

def viterbi_output(dev_out_path, X_raw, X, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum):
    
    index2tag = {value: key for key, value in tag2index.items()}
    Y = [viterbi(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum) for x in X]
    tags = [idx2tag(y, index2tag) for y in Y]
    
    output_string = ''
    for i in range(len(X)):
        for j in range(len(X[i])):
            output_string += X_raw[i][j][0] + ' ' + X_raw[i][j][1] + ' ' + tags[i][j] + '\n'
        output_string += '\n'
    
    with open(dev_out_path, 'w') as f:
        f.write(output_string)
    
    print('Done with writing predictions')
    return None

def eval(X, Y_raw, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum):
    index2tag = {value: key for key, value in tag2index.items()}
    def flatten(L):
        return [e for l in L for e in l]
    Y_pred = flatten([idx2tag(viterbi(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum), index2tag) for x in X])
    Y_raw  = flatten(Y_raw)
    assert len(Y_raw) == len(Y_pred)
    return evaluate(Y_raw, Y_pred)

def Loss(X, Y, tag2index, word2index, postag2index, transition_weight, emission_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum):
    
    loss = 0
    
    for x, y in zip(X, Y):
        transition_count, emission_count, emission_count_pos, combination_count, combination_count_pos = feature_count(x, y, tag2index, word2index, postag2index)
        loss -= (np.sum(transition_count*transition_weight) + np.sum(emission_count*emission_weight) + np.sum(emission_count_pos*emission_weight_pos) + np.sum(combination_count*combination_weight) + np.sum(combination_count_pos*combination_weight_pos))
        y_pred = viterbi(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum)
        transition_count_pred, emission_count_pred, emission_count_pos_pred, combination_count_pred, combination_count_pos_pred = feature_count(x, y_pred, tag2index, word2index, postag2index)
        loss += (np.sum(transition_count_pred*transition_weight) + np.sum(emission_count_pred*emission_weight) + np.sum(emission_count_pos_pred*emission_weight_pos) + np.sum(combination_count_pred*combination_weight) + np.sum(combination_count_pos_pred*combination_weight_pos))
    
    return loss

def feature_count(x, y, tag2index, word2index, postag2index):
    
    T, V, POS, N = len(tag2index), len(word2index), len(postag2index), len(x)
    transition_count, emission_count, emission_count_pos, combination_count, combination_count_pos = np.zeros((T+1, T+1)), np.zeros((T, V)), np.zeros((T, POS)), np.zeros((T+1, T, V)), np.zeros((T+1, T, POS))

    transition_count[-1, y[0]] += 1
    for i in range(1, N):
        transition_count[y[i-1], y[i]] += 1
    transition_count[y[-1], -1] += 1

    for i in range(N):
        emission_count[y[i], x[i][0]] += 1
        emission_count_pos[y[i], x[i][1]] += 1

    combination_count[-1, y[0], x[0][0]] += 1
    combination_count_pos[-1, y[0], x[0][1]] += 1
    for i in range(1, N):
        combination_count[y[i-1], y[i], x[i][0]] += 1
        combination_count_pos[y[i-1], y[i], x[i][1]] += 1

    return transition_count, emission_count, emission_count_pos, combination_count, combination_count_pos

def trainDecay(X, Y, X_dev, Y_dev_raw, tag2index, word2index, postag2index, link_weight_sum, iteration=20, random_seed=1):

    T, V, POS, D = len(tag2index), len(word2index), len(postag2index), len(X)
    f1_opti, counter = 0, 1
    transition_weight, emission_weight, emission_weight_pos, combination_weight, combination_weight_pos = np.zeros((T+1, T+1)), np.zeros((T, V)), np.zeros((T, POS)), np.zeros((T+1, T, V)), np.zeros((T+1, T, POS))
    np.random.seed(random_seed)

    for i in range(iteration):
        for j in range(D):
            k = random.randint(0, D-1)
            x, y = X[k], Y[k]
            transition_count, emission_count, emission_count_pos, combination_count, combination_count_pos = feature_count(x, y, tag2index, word2index, postag2index)
            y_pred = viterbi(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum)
            transition_count_pred, emission_count_pred, emission_count_pos_pred, combination_count_pred, combination_count_pos_pred = feature_count(x, y_pred, tag2index, word2index, postag2index)
            transition_weight += (transition_count - transition_count_pred)*1/counter
            emission_weight += (emission_count - emission_count_pred)*1/counter
            emission_weight_pos += (emission_count_pos - emission_count_pos_pred)*1/counter
            combination_weight += (combination_count - combination_count_pred)*1/counter
            combination_weight_pos += (combination_count_pos - combination_count_pos_pred)*1/counter
        loss = Loss(X, Y, tag2index, word2index, postag2index, transition_weight, emission_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum)
        with HiddenPrints():
            prec, rec, f1 = eval(X_dev, Y_dev_raw, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum)
        print('training epoch: {} , training loss: {:.4f} dev F1: {:.4f}'.format(i+1, loss, f1))
        counter += 1
        if f1 > f1_opti:
            transition_weight_opti, emission_weight_opti, emission_weight_pos_opti, combination_weight_opti, combination_weight_pos_opti, f1_opti = transition_weight.copy(), emission_weight.copy(), emission_weight_pos.copy(), combination_weight.copy(), combination_weight_pos.copy(), f1
            print('better parameters found!')

    return transition_weight_opti, emission_weight_opti, emission_weight_pos_opti, combination_weight_opti, combination_weight_pos_opti

def main():
    path = Path('../data/full')
    train_X, train_X_str, train_Y, train_Y_str, dev_X, dev_X_str, dev_Y, dev_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, postag2index, index2tag = get_xy(path)

    print('************Training Set Summary*************')
    T, V, POS = len(tag2index), len(word2index), len(postag2index)
    print('Number of tags: {}, Number of words: {}, Number of pos tags: {}'.format(T, V, POS))

    print('************Train*************')
    start = time.time()
    optimal_transition_weight, optimal_emission_weight, optimal_emission_weight_pos, optimal_combination_weight, optimal_combination_weight_pos = trainDecay(train_X, train_Y, dev_X, dev_Y_str, tag2index, word2index, postag2index, link_weight_sum, iteration=20, random_seed=1)
    end = time.time()
    time_elapsed(start, end)
    
    # print('************Saving Model Parameters*************')
    # path_transition = path/'best_weight_features3_transition_strp.npy'
    # path_emission = path/'best_weight_features3_emission_strp.npy'
    # np.save(path_transition, optimal_transition_weight)
    # np.save(path_emission, optimal_emission_weight)
    
    print('************Evaluation*************')
    prec, rec, f1 = eval(train_X, train_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, optimal_emission_weight_pos, optimal_combination_weight, optimal_combination_weight_pos, link_weight_sum)
    print('precision, recall, f1 on training set: {0} {1} {2}'.format(prec, rec, f1))
    prec, rec, f1 = eval(dev_X, dev_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, optimal_emission_weight_pos, optimal_combination_weight, optimal_combination_weight_pos, link_weight_sum)
    print('precision, recall, f1 on development set: {0} {1} {2}'.format(prec, rec, f1))
    prec, rec, f1 = eval(test_X, test_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, optimal_emission_weight_pos, optimal_combination_weight, optimal_combination_weight_pos, link_weight_sum)
    print('precision, recall, f1 on test set: {0} {1} {2}'.format(prec, rec, f1))


if __name__=='__main__':
    main()