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
        for line in f.readlines():
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
    # Expanded for testing small number of instances
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
    train_ds, devin_ds, devout_ds = read_data(path)
    train_X, train_X_str, train_Y, train_Y_str, word2index, tag2index, postag2index, index2tag = idx_xy(train_ds[0], train_ds[1])
    # test_X, test_X_str, test_Y, test_Y_str, _, _, _, _ = idx_xy(devout_ds[0], devout_ds[1])
    test_X, test_X_str, test_Y, test_Y_str, _, _, _, _ = idx_xy(devout_ds[0], devout_ds[1], word2index, tag2index, postag2index)
    return train_X, train_X_str, train_Y, train_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, postag2index, index2tag

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

def viterbi(X, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum):
    
    Y = list()
    index2tag = {value: key for key, value in tag2index.items()}

    for x in X:
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
        path = [last_idx]
        for m in range(len(x)-1, 0, -1):
            path.insert(0, path_matrix[path[0], m])
        
        Y.append([index2tag[idx] for idx in path])
    
    return Y

def viterbi_output(dev_out_path, X_raw, X, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum):
    
    tags = viterbi(X, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum)
    
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
    Y_pred = flatten(viterbi(X, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, link_weight_sum))
    Y_raw  = flatten(Y_raw)
    assert len(Y_raw) == len(Y_pred)
    return evaluate(Y_raw, Y_pred)

def forward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos):
    N, T = len(x), len(tag2index)
    forward_matrix = np.zeros((T, N), dtype=np.double)

    forward_matrix[:, 0] = transition_weight[-1, :-1] + emission_weight_pos[:, x[0][1]] + emission_weight[:, x[0][0]] + combination_weight[-1, :, x[0][0]] + combination_weight_pos[-1, :, x[0][1]]
    for i in range(1, N): 
        forward_matrix[:, i] = logsumexp(forward_matrix[:, i-1][:, None] + link_weight_sum(x[i], transition_weight, emission_weight, emission_weight_pos, combination_weight, combination_weight_pos), axis=0)
    log_Z = logsumexp(transition_weight[:-1, -1] + forward_matrix[:, -1])
    
    return forward_matrix, log_Z

def backward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos):
    N, T = len(x), len(tag2index)
    backward_matrix = np.zeros((T, N), dtype=np.double)
    
    backward_matrix[:, -1] = transition_weight[:-1, -1]
    for i in range(N-2, -1, -1):
        backward_matrix[:, i] = logsumexp(np.expand_dims(backward_matrix[:, i+1], axis=0) + link_weight_sum(x[i+1], transition_weight, emission_weight, emission_weight_pos, combination_weight, combination_weight_pos), axis=1)
    log_Z = logsumexp(transition_weight[-1, :-1] + emission_weight[:, x[0][0]] + emission_weight_pos[:, x[0][1]] + combination_weight[-1, :, x[0][0]] + combination_weight_pos[-1, :, x[0][1]] + backward_matrix[:, 0])
    return backward_matrix, log_Z

def Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):
    
    loss = 0
    
    for x, y in zip(X, Y):
        pair_score = 0
        emission_score = emission_weight[y[0], x[0][0]] + emission_weight_pos[y[0], x[0][1]]
        transition_score = transition_weight[-1, y[0]]
        combination_score = combination_weight[-1, y[0], x[0][0]] + combination_weight_pos[-1, y[0], x[0][1]]
        pair_score += (transition_score + emission_score + combination_score)
        for i in range(1, len(x)):
            emission_score = emission_weight[y[i], x[i][0]] + emission_weight_pos[y[i], x[i][1]]
            transition_score = transition_weight[y[i-1], y[i]]
            combination_score = combination_weight[y[i-1], y[i], x[i][0]] + combination_weight_pos[y[i-1], y[i], x[i][1]]
            pair_score += (transition_score + emission_score + combination_score)
        
        transition_score = transition_weight[y[-1], -1]
        pair_score += transition_score
        
        _, log_Z = forward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)

        loss += -(pair_score - log_Z)
    
    loss += LossRegularization(emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
    
    return loss

def LossRegularization(emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):
    return param*(np.sum(emission_weight[emission_weight != -np.inf]**2) +\
            np.sum(transition_weight[transition_weight != -np.inf]**2) +\
            np.sum(emission_weight_pos[emission_weight_pos != -np.inf]**2) +\
            np.sum(combination_weight[combination_weight != -np.inf]**2) +\
            np.sum(combination_weight_pos[combination_weight_pos != -np.inf]**2))

def GradientTransition(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):
    
    T = len(tag2index)
    counter = 1
    Expected_count, Empirical_count = np.zeros((T+1, T+1), dtype=np.double), np.zeros((T+1, T+1))
    
    for x, y in zip(X, Y):
        N = len(x)
        forward_matrix, log_Z = forward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
        backward_matrix, _ = backward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)

        expected_count, empirical_count = np.zeros((T+1, T+1), dtype=np.double), np.zeros((T+1, T+1))

        for tag1, tag2 in itertools.product(range(-1, T), range(-1, T)):
            log_SumPotential = 0

            transition_score = transition_weight[tag1, tag2]
            if tag1 == -1 and tag2 == -1:
                continue
                # both empirical and expected count set to 0, forcing the parameter to go to 0 becasue of L2 regularization, which doesn't matter at all
            elif tag1 == -1:
                emission_score = emission_weight[tag2, x[0][0]] + emission_weight_pos[tag2, x[0][1]]
                combination_score = combination_weight[tag1, tag2, x[0][0]] + combination_weight_pos[tag1, tag2, x[0][1]]
                log_SumPotential += transition_score + emission_score + combination_score + backward_matrix[tag2, 0]
            elif tag2 == -1:
                log_SumPotential += forward_matrix[tag1, -1] + transition_score
            else:
                log_SumPotential += logsumexp(forward_matrix[tag1, :N-1] + transition_score + emission_weight[tag2, [pair[0] for pair in x[1:N]]] + emission_weight_pos[tag2, [pair[1] for pair in x[1:N]]] +\
                                              combination_weight[tag1, tag2, [pair[0] for pair in x[1:N]]] + combination_weight_pos[tag1, tag2, [pair[1] for pair in x[1:N]]] + backward_matrix[tag2, 1:N])
    
            expected_count[tag1, tag2] = np.exp(log_SumPotential - log_Z)
            
        Expected_count += expected_count

        empirical_count[-1, y[0]] += 1
        for i in range(N-1):
            empirical_count[y[i], y[i+1]] += 1
        empirical_count[y[-1], -1] += 1

        Empirical_count += empirical_count

        if counter % 100 == 0:
            print('Transition: done with the {}th instances'.format(counter))
        counter += 1

    L2_gradient = 2*param*transition_weight
    L2_gradient[L2_gradient == -np.inf] = 0
    
    return Expected_count - Empirical_count + L2_gradient

def GradientEmission(X, Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):

    T = len(tag2index)
    V = len(word2index)
    counter = 1
    Expected_count, Empirical_count = np.zeros((T, V), dtype=np.double), np.zeros((T, V))
    
    for x, y in zip(X, Y):
        N = len(x)
        forward_matrix, log_Z = forward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
        backward_matrix, _ = backward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
    
        expected_count = np.zeros((T, V), dtype=np.double)

        emission_score = emission_weight[:, x[0][0]] + emission_weight_pos[:, x[0][1]]
        combination_score = combination_weight[-1, :, x[0][0]] + combination_weight_pos[-1, :, x[0][1]]
        transition_score = transition_weight[-1, :-1]
        expected_count[:, x[0][0]] += np.exp(backward_matrix[:, 0] + emission_score + transition_score + combination_score - log_Z)
        
        for i in range(1, N):
            emission_score = emission_weight[:, x[i][0]] + emission_weight_pos[:, x[i][1]]
            combination_scores = combination_weight[:-1, :, x[i][0]] + combination_weight_pos[:-1, :, x[i][1]]
            transition_scores = transition_weight[:-1, :-1]
            expected_count[:, x[i][0]] += np.sum(np.exp(forward_matrix[:, i-1][:, None] + np.expand_dims(backward_matrix[:, i] + emission_score, axis=0) + transition_scores + combination_scores - log_Z), axis=0)

        Expected_count += expected_count

        for pair, tag in zip(x, y):
            Empirical_count[tag, pair[0]] += 1

        if counter % 100 == 0:
            print('Emission: done with the {}th instances'.format(counter))
        counter += 1

    L2_gradient = 2*param*emission_weight
    L2_gradient[L2_gradient == -np.inf] = 0

    return Expected_count - Empirical_count + L2_gradient

def GradientEmissionPOS(X, Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):

    T = len(tag2index)
    POS = len(postag2index)
    counter = 1
    Expected_count, Empirical_count = np.zeros((T, POS), dtype=np.double), np.zeros((T, POS))
    
    for x, y in zip(X, Y):
        N = len(x)
        forward_matrix, log_Z = forward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
        backward_matrix, _ = backward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
    
        expected_count = np.zeros((T, POS), dtype=np.double)

        emission_score = emission_weight[:, x[0][0]] + emission_weight_pos[:, x[0][1]]
        combination_score = combination_weight[-1, :, x[0][0]] + combination_weight_pos[-1, :, x[0][1]]
        transition_score = transition_weight[-1, :-1]
        expected_count[:, x[0][1]] += np.exp(backward_matrix[:, 0] + emission_score + transition_score + combination_score - log_Z)

        for i in range(1, N):
            emission_score = emission_weight[:, x[i][0]] + emission_weight_pos[:, x[i][1]]
            combination_scores = combination_weight[:-1, :, x[i][0]] + combination_weight_pos[:-1, :, x[i][1]]
            transition_scores = transition_weight[:-1, :-1]
            expected_count[:, x[i][1]] += np.sum(np.exp(forward_matrix[:, i-1][:, None] + np.expand_dims(backward_matrix[:, i] + emission_score, axis=0) + transition_scores + combination_scores - log_Z), axis=0)
        
        Expected_count += expected_count

        for pair, tag in zip(x, y):
            Empirical_count[tag, pair[1]] += 1

        if counter % 100 == 0:
            print('Emission POS: done with the {}th instances'.format(counter))
        counter += 1

    L2_gradient = 2*param*emission_weight_pos
    L2_gradient[L2_gradient == -np.inf] = 0

    return Expected_count - Empirical_count + L2_gradient  

def GradientCombination(X, Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):

    T = len(tag2index)
    V = len(word2index)
    counter = 1
    Expected_count, Empirical_count = np.zeros((T+1, T, V), dtype=np.double), np.zeros((T+1, T, V))
    
    for x, y in zip(X, Y):
        N = len(x)
        forward_matrix, log_Z = forward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
        backward_matrix, _ = backward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
    
        expected_count = np.zeros((T+1, T, V), dtype=np.double)


        emission_score = emission_weight[:, x[0][0]] + emission_weight_pos[:, x[0][1]]
        combination_score = combination_weight[-1, :, x[0][0]] + combination_weight_pos[-1, :, x[0][1]]
        transition_score = transition_weight[-1, :-1]
        expected_count[-1, :, x[0][0]] += np.exp(backward_matrix[:, 0] + emission_score + transition_score + combination_score - log_Z)
        
        for i in range(1, N):
            emission_score = emission_weight[:, x[i][0]] + emission_weight_pos[:, x[i][1]]
            combination_scores = combination_weight[:-1, :, x[i][0]] + combination_weight_pos[:-1, :, x[i][1]]
            transition_scores = transition_weight[:-1, :-1]
            expected_count[:-1, :, x[i][0]] += np.exp(forward_matrix[:, i-1][:, None] + np.expand_dims(backward_matrix[:, i] + emission_score, axis=0) + transition_scores + combination_scores - log_Z)
        
        Expected_count += expected_count

        Empirical_count[-1, y[0], x[0][0]] += 1
        for i in range(1, N):
            Empirical_count[y[i-1], y[i], x[i][0]] += 1

        if counter % 100 == 0:
            print('Combination: done with the {}th instances'.format(counter))
        counter += 1

    L2_gradient = 2*param*combination_weight
    L2_gradient[L2_gradient == -np.inf] = 0

    return Expected_count - Empirical_count + L2_gradient

def GradientCombinationPOS(X, Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):

    T = len(tag2index)
    POS = len(postag2index)
    counter = 1
    Expected_count, Empirical_count = np.zeros((T+1, T, POS), dtype=np.double), np.zeros((T+1, T, POS))
    
    for x, y in zip(X, Y):
        N = len(x)
        forward_matrix, log_Z = forward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
        backward_matrix, _ = backward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
    
        expected_count = np.zeros((T+1, T, POS), dtype=np.double)


        emission_score = emission_weight[:, x[0][0]] + emission_weight_pos[:, x[0][1]]
        combination_score = combination_weight[-1, :, x[0][0]] + combination_weight_pos[-1, :, x[0][1]]
        transition_score = transition_weight[-1, :-1]
        expected_count[-1, :, x[0][1]] += np.exp(backward_matrix[:, 0] + emission_score + transition_score + combination_score - log_Z)
        
        for i in range(1, N):
            emission_score = emission_weight[:, x[i][0]] + emission_weight_pos[:, x[i][1]]
            combination_scores = combination_weight[:-1, :, x[i][0]] + combination_weight_pos[:-1, :, x[i][1]]
            transition_scores = transition_weight[:-1, :-1]
            expected_count[:-1, :, x[i][1]] += np.exp(forward_matrix[:, i-1][:, None] + np.expand_dims(backward_matrix[:, i] + emission_score, axis=0) + transition_scores + combination_scores - log_Z)
        
        Expected_count += expected_count

        Empirical_count[-1, y[0], x[0][1]] += 1
        for i in range(1, N):
            Empirical_count[y[i-1], y[i], x[i][1]] += 1

        if counter % 100 == 0:
            print('Combination POS: done with the {}th instances'.format(counter))
        counter += 1

    L2_gradient = 2*param*combination_weight_pos
    L2_gradient[L2_gradient == -np.inf] = 0

    return Expected_count - Empirical_count + L2_gradient  

def testTransitionAll(epsilon, X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):
    T = len(tag2index)
    result_loss_dff_actual = np.zeros((T+1, T+1))
    result_loss_dff_predicted = np.zeros((T+1, T+1))
    with HiddenPrints():
        gradient = GradientTransition(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        result_loss_dff_predicted = gradient*epsilon
        old_loss = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        for tag1, tag2 in itertools.product(range(-1, T), range(-1, T)):
            transition_weight[tag1, tag2] += epsilon
            result_loss_dff_actual[tag1, tag2] = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param) - old_loss
            transition_weight[tag1, tag2] -= epsilon
        difference = np.abs(result_loss_dff_actual - result_loss_dff_predicted)
    print('********transition********')
    print('difference: {}, argmax: {}, max: {}'.format(difference, np.argmax(difference), np.max(difference)))  

def testEmissionAll(epsilon, X, Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):
    T, V = len(tag2index), len(word2index)
    result_loss_dff_actual = np.zeros((T, V))
    result_loss_dff_predicted = np.zeros((T, V))
    with HiddenPrints():
        gradient = GradientEmission(X, Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        result_loss_dff_predicted = gradient*epsilon
        old_loss = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        for tag, word in itertools.product(range(T), range(V)):
            emission_weight[tag, word] += epsilon
            result_loss_dff_actual[tag, word] = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param) - old_loss
            emission_weight[tag, word] -= epsilon                                        
        difference = np.abs(result_loss_dff_actual - result_loss_dff_predicted)
    print('********emission********')
    print('difference: {}, argmax: {}, max: {}'.format(difference, np.argpartition(difference, -10, axis=None)[-10:], np.partition(difference, -10, axis=None)[-10:]))

def testEmissionPOSAll(epsilon, X, Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):
    T, POS = len(tag2index), len(postag2index)
    result_loss_dff_actual = np.zeros((T, POS))
    result_loss_dff_predicted = np.zeros((T, POS))
    with HiddenPrints():
        gradient = GradientEmissionPOS(X, Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        result_loss_dff_predicted = gradient*epsilon
        old_loss = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        for tag, postag in itertools.product(range(T), range(POS)):
            emission_weight_pos[tag, postag] += epsilon
            result_loss_dff_actual[tag, postag] = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param) - old_loss
            emission_weight_pos[tag, postag] -= epsilon                                        
        difference = np.abs(result_loss_dff_actual - result_loss_dff_predicted)
    print('********emission pos********')
    print('difference: {}, argmax: {}, max: {}'.format(difference, np.argpartition(difference, -10, axis=None)[-10:], np.partition(difference, -10, axis=None)[-10:]))

def testCombinationAll(epsilon, X, Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):
    T, V = len(tag2index), len(word2index)
    result_loss_dff_actual = np.zeros((T+1, T, V))
    result_loss_dff_predicted = np.zeros((T+1, T, V))
    with HiddenPrints():
        gradient = GradientCombination(X, Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        result_loss_dff_predicted = gradient*epsilon
        old_loss = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        for tag1, tag2, word in itertools.product(range(T+1), range(T), range(V)):
            combination_weight[tag1, tag2, word] += epsilon
            result_loss_dff_actual[tag1, tag2, word] = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param) - old_loss
            combination_weight[tag1, tag2, word] -= epsilon                                        
        difference = np.abs(result_loss_dff_actual - result_loss_dff_predicted)
    print('********combination********')
    print('difference: {}, argmax: {}, max: {}'.format(difference, np.argpartition(difference, -10, axis=None)[-10:], np.partition(difference, -10, axis=None)[-10:]))

def testCombinationPOSAll(epsilon, X, Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param):
    T, POS = len(tag2index), len(postag2index)
    result_loss_dff_actual = np.zeros((T+1, T, POS))
    result_loss_dff_predicted = np.zeros((T+1, T, POS))
    with HiddenPrints():
        gradient = GradientCombinationPOS(X, Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        result_loss_dff_predicted = gradient*epsilon
        old_loss = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param)
        for tag1, tag2, postag in itertools.product(range(T+1), range(T), range(POS)):
            combination_weight_pos[tag1, tag2, postag] += epsilon
            result_loss_dff_actual[tag1, tag2, postag] = Loss(X, Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param) - old_loss
            combination_weight_pos[tag1, tag2, postag] -= epsilon                                        
        difference = np.abs(result_loss_dff_actual - result_loss_dff_predicted)
    print('********combination pos********')
    print('difference: {}, argmax: {}, max: {}'.format(difference, np.argpartition(difference, -10, axis=None)[-10:], np.partition(difference, -10, axis=None)[-10:]))

def forward_backward_test(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos):

    _, log_Z1 = forward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
    _, log_Z2 = backward(x, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
    print(log_Z1, log_Z2)

def main():
    path = Path('../data/full')
    train_X, train_X_str, train_Y, train_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, postag2index, index2tag = get_xy(path)

    print('************Training Set Summary*************')
    T, V, POS = len(tag2index), len(word2index), len(postag2index)
    print('Number of tags: {}, Number of words: {}, Number of POS tags: {}'.format(T, V, POS))

    # print('************Test Forward Backward*************')
    # np.random.seed(1)
    # emission_weight = np.random.rand(T, V)
    # transition_weight = np.random.rand(T+1, T+1)
    # emission_weight_pos = np.random.rand(T, POS)
    # combination_weight = np.random.rand(T+1, T, V)
    # combination_weight_pos = np.random.rand(T+1, T, POS)
    # forward_backward_test(train_X[0], tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)
    # forward_backward_test(train_X[-1], tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos)

    # print('************Test Perturbation*************')
    # epsilon = 0.00000001
    # testTransitionAll(epsilon, train_X, train_Y, tag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param=0.0)
    # print('training X: ', train_X)
    # testEmissionAll(epsilon, train_X, train_Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param=0.0)
    # testEmissionPOSAll(epsilon, train_X, train_Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param=0.0)
    # testCombinationAll(epsilon, train_X, train_Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param=0.0)
    # testCombinationPOSAll(epsilon, train_X, train_Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param=0.0)
    
    # GradientEmission(train_X, train_Y, tag2index, word2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param=0.0)
    # GradientEmissionPOS(train_X, train_Y, tag2index, postag2index, emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param=0.0)
    
    Lambda = 0.1

    def callbackF(w):
        loss = get_loss_grad(w)[0]
        transition_weight = w[:(T+1)*(T+1)].reshape((T+1, T+1))
        emission_weight = w[(T+1)*(T+1):(T+1)*(T+1)+T*V].reshape((T, V))
        emission_weight_pos = w[(T+1)*(T+1)+T*V:(T+1)*(T+1)+T*(V+POS)].reshape((T, POS))
        combination_weight = w[(T+1)*(T+1)+T*(V+POS):(T+1)*(T+1)+T*(V+POS)+(T+1)*T*V].reshape((T+1, T, V))
        combination_weight_pos = w[(T+1)*(T+1)+T*(V+POS)+(T+1)*T*V:].reshape((T+1, T, POS))
        loss_l2 = LossRegularization(emission_weight, transition_weight, emission_weight_pos, combination_weight, combination_weight_pos, param=Lambda)
        print('Loss:{:.4f} L2 Loss:{:.4f}'.format(loss, loss_l2))

    def get_loss_grad(w):
        with HiddenPrints():
            transition_weight = w[:(T+1)*(T+1)].reshape((T+1, T+1))
            emission_weight = w[(T+1)*(T+1):(T+1)*(T+1)+T*V].reshape((T, V))
            emission_weight_pos = w[(T+1)*(T+1)+T*V:(T+1)*(T+1)+T*(V+POS)].reshape((T, POS))
            combination_weight = w[(T+1)*(T+1)+T*(V+POS):(T+1)*(T+1)+T*(V+POS)+(T+1)*T*V].reshape((T+1, T, V))
            combination_weight_pos = w[(T+1)*(T+1)+T*(V+POS)+(T+1)*T*V:].reshape((T+1, T, POS))
            loss = Loss(train_X, train_Y, tag2index, 
                        emission_weight, transition_weight, emission_weight_pos, 
                        combination_weight, combination_weight_pos, param=Lambda)
            grads_transition = GradientTransition(train_X, train_Y, tag2index,
                                                emission_weight, transition_weight, emission_weight_pos, 
                                                combination_weight, combination_weight_pos, param=Lambda)
            grads_emission = GradientEmission(train_X, train_Y, tag2index, word2index, 
                                            emission_weight, transition_weight, emission_weight_pos, 
                                            combination_weight, combination_weight_pos, param=Lambda)
            grads_emission_pos = GradientEmissionPOS(train_X, train_Y, tag2index, postag2index, 
                                            emission_weight, transition_weight, emission_weight_pos, 
                                            combination_weight, combination_weight_pos, param=Lambda)
            grads_combination = GradientCombination(train_X, train_Y, tag2index, word2index, 
                                            emission_weight, transition_weight, emission_weight_pos, 
                                            combination_weight, combination_weight_pos, param=Lambda)
            grads_combination_pos = GradientCombinationPOS(train_X, train_Y, tag2index, postag2index, 
                                            emission_weight, transition_weight, emission_weight_pos, 
                                            combination_weight, combination_weight_pos, param=Lambda)
            grads = np.concatenate((grads_transition.reshape(-1), grads_emission.reshape(-1), grads_emission_pos.reshape(-1), grads_combination.reshape(-1), grads_combination_pos.reshape(-1)))
        return loss, grads

    print('************Train*************')
    start = time.time()
    init_w = np.zeros(((T+1)*(T+1)+T*(V+POS)+T*(T+1)*(V+POS),))
    optimal_weight, final_loss, result_dict = fmin_l_bfgs_b(get_loss_grad, init_w, pgtol=0.01, callback=callbackF)
    end = time.time()
    time_elapsed(start, end)
    
    print('************Saving Model Parameters*************')
    optimal_transition_weight = optimal_weight[:(T+1)*(T+1)].reshape((T+1, T+1))
    optimal_emission_weight = optimal_weight[(T+1)*(T+1):(T+1)*(T+1)+T*V].reshape((T, V))
    optimal_emission_weight_pos = optimal_weight[(T+1)*(T+1)+T*V:(T+1)*(T+1)+T*(V+POS)].reshape((T, POS))
    optimal_combination_weight = optimal_weight[(T+1)*(T+1)+T*(V+POS):(T+1)*(T+1)+T*(V+POS)+(T+1)*T*V].reshape((T+1, T, V))
    optimal_combination_weight_pos = optimal_weight[(T+1)*(T+1)+T*(V+POS)+(T+1)*T*V:].reshape((T+1, T, POS))
    path_transition = path/'best_weight_features3_transition.npy'
    path_emission = path/'best_weight_features3_emission.npy'
    path_emission_pos  = path/'best_weight_features3_emission_pos.npy'
    path_combination = path/'best_weight_features3_combination.npy'
    path_combination_pos = path/'best_weight_features3_combination_pos.npy'
    np.save(path_transition, optimal_transition_weight)
    np.save(path_emission, optimal_emission_weight)
    np.save(path_emission_pos, optimal_emission_weight_pos)
    np.save(path_combination, optimal_combination_weight)
    np.save(path_combination_pos, optimal_combination_weight_pos)

    print('************Saving Model Outputs*************')
    path_output = path/'dev.p5.CRF.f4.out'
    viterbi_output(path_output, test_X_str, test_X, tag2index, optimal_emission_weight, optimal_transition_weight, optimal_emission_weight_pos, optimal_combination_weight, optimal_combination_weight_pos, link_weight_sum)
    
    print('************Evaluation*************')
    prec, rec, f1 = eval(train_X, train_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, optimal_emission_weight_pos, optimal_combination_weight, optimal_combination_weight_pos, link_weight_sum)
    print('precision, recall, f1 on training set: {0} {1} {2}'.format(prec, rec, f1))
    prec, rec, f1 = eval(test_X, test_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, optimal_emission_weight_pos, optimal_combination_weight, optimal_combination_weight_pos, link_weight_sum)
    print('precision, recall, f1 on test set: {0} {1} {2}'.format(prec, rec, f1))


if __name__=='__main__':
    main()