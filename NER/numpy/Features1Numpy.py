import numpy as np
from pathlib import Path
from scipy.optimize import fmin_l_bfgs_b
import itertools
from conlleval_ import evaluate
import copy
import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

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
    test_X, test_X_str, test_Y, test_Y_str, _, _, _ = idx_xy(devout_ds[0], devout_ds[1])
    return train_X, train_X_str, train_Y, train_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, index2tag

############### Part 1 ###############
def transition_weight(Y, tag2index):
    
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

def emission_weight(X, Y, word2index, tag2index):
    
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

def link_weight_product(x, transition_weight_exp, emission_weight_exp):
    """
    The -1 check remains in case it is used for calculation of loss on a new test set
    """
    T = transition_weight_exp.shape[0] - 1 
    emission = np.ones((1, T))

    if x != -1:
        emission = np.expand_dims(emission_weight_exp[:, x], axis=0)
    transition = transition_weight_exp[:-1, :-1]
    return transition * emission

def forward(x, tag2index, emission_weight, transition_weight):
    """
    The returned value should be the forward matrix for sentence X and the final alpha
    The implementation so far explicitly pass emission and transition dictionary as arguments
    This can be replaced by FeatureSum function in the implementation for part 5
    """
    N, T = len(x), len(tag2index)
    forward_matrix = np.zeros((T, N))
    emission_weight_exp = np.exp(emission_weight)
    transition_weight_exp = np.exp(transition_weight)

    forward_matrix[:, 0] = transition_weight_exp[-1, :-1]*emission_weight_exp[:, x[0]] if x[0] != -1 else transition_weight_exp[-1, :-1]
    for i in range(1, N): 
        forward_matrix[:, i] = np.sum(forward_matrix[:, i-1][:, None]*link_weight_product(x[i], transition_weight_exp, emission_weight_exp), axis=0)
    
    SumPotential = np.sum(transition_weight_exp[:-1, -1]*forward_matrix[:, -1])
    
    return forward_matrix, SumPotential

def forwardNew(x, tag2index, emission_weight, transition_weight):
    """
    To avoid overflow issue in case there is
    """
    N, T = len(x), len(tag2index)
    forward_matrix = np.zeros((T, N))

    forward_matrix[:, 0] = transition_weight[-1, :-1]+emission_weight[:, x[0]] if x[0] != -1 else transition_weight[-1, :-1]
    for i in range(1, N): 
        forward_matrix[:, i] = np.log(np.sum(np.exp(forward_matrix[:, i-1][:, None] + link_weight_sum(x[i], transition_weight, emission_weight)), axis=0))
    
    SumPotential = np.sum(np.exp(transition_weight[:-1, -1] + forward_matrix[:, -1]))
    
    return np.exp(forward_matrix), SumPotential

def backward(x, tag2index, emission_weight, transition_weight):
    """
    The returned value should be the forward matrix for sentence X and the final beta
    final beta should be the same as final alpha
    This is considered correct according to the definition of backward algorithm
    """
    N, T = len(x), len(tag2index)
    backward_matrix = np.zeros((T, N))
    emission_weight_exp = np.exp(emission_weight)
    transition_weight_exp = np.exp(transition_weight)
    
    backward_matrix[:, -1] = transition_weight_exp[:-1, -1]
    for i in range(N-2, -1, -1):
        backward_matrix[:, i] = np.sum(np.expand_dims(backward_matrix[:, i+1], axis=0)*link_weight_product(x[i+1], transition_weight_exp, emission_weight_exp), axis=1)
    
    SumPotential = np.sum(transition_weight_exp[-1, :-1]*emission_weight_exp[:, x[0]]*backward_matrix[:, 0])
    
    return backward_matrix, SumPotential

def backwardNew(x, tag2index, emission_weight, transition_weight):
    """
    To avoid overflow issue in case there is
    """
    N, T = len(x), len(tag2index)
    backward_matrix = np.zeros((T, N))
    
    backward_matrix[:, -1] = transition_weight[:-1, -1]
    for i in range(N-2, -1, -1):
        backward_matrix[:, i] = np.log(np.sum(np.exp(np.expand_dims(backward_matrix[:, i+1], axis=0) + link_weight_sum(x[i+1], transition_weight, emission_weight)), axis=1))
    
    SumPotential = np.sum(np.exp(transition_weight[-1, :-1] + emission_weight[:, x[0]] + backward_matrix[:, 0]))
    
    return np.exp(backward_matrix), SumPotential

def Loss(X, Y, tag2index, emission_weight, transition_weight, param=0.1):
    
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
        
        _, SumPotential = forward(x, tag2index, emission_weight, transition_weight)

        loss += -(pair_score-np.log(SumPotential))
    
    loss += param*(np.sum(emission_weight[emission_weight != -np.inf]**2) +\
            np.sum(transition_weight[transition_weight != -np.inf]**2))
    
    return loss 

def GradientTransition(X, Y, tag2index, word2index, emission_weight, transition_weight, param=0.1):
    
    T = len(tag2index)
    counter = 1
    Expected_count, Empirical_count = np.zeros((T+1, T+1)), np.zeros((T+1, T+1))
    
    for x, y in zip(X, Y):
        N = len(x)
        forward_matrix, NormalizationTerm = forward(x, tag2index, emission_weight, transition_weight)
        backward_matrix, _ = backward(x, tag2index, emission_weight, transition_weight)

        expected_count, empirical_count = np.zeros((T+1, T+1)), np.zeros((T+1, T+1))

        for tag1, tag2 in itertools.product(range(-1, T), range(-1, T)):
            SumPotential = 0

            transition_score = np.exp(transition_weight[tag1, tag2])
            if tag1 == -1 and tag2 == -1:
                continue 
                # both empirical and expected count set to 0, forcing the parameter to go to 0 becasue of L2 regularization, which doesn't matter at all
            elif tag1 == -1:
                emission_score = np.exp(emission_weight[tag2, x[0]])
                SumPotential += transition_score*emission_score*backward_matrix[tag2, 0]
            elif tag2 == -1:
                SumPotential += forward_matrix[tag1, -1]*transition_score
            else:
                for i in range(N-1):
                    emission_score = np.exp(emission_weight[tag2, x[i+1]])
                    SumPotential += forward_matrix[tag1, i]*transition_score*emission_score*backward_matrix[tag2, i+1]
                    
            expected_count[tag1, tag2] = SumPotential/NormalizationTerm
            
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

def GradientEmissionOld(X, Y, tag2index, word2index, emission_weight, transition_weight, param=0.1):
    """
    The double for loop can be further optimized a lot
    """
    T = len(tag2index)
    V = len(word2index)
    counter = 1
    Expected_count, Empirical_count = np.zeros((T, V)), np.zeros((T, V))
    
    for x, y in zip(X, Y):
        N = len(x)
        forward_matrix, NormalizationTerm = forward(x, tag2index, emission_weight, transition_weight)
        backward_matrix, _ = backward(x, tag2index, emission_weight, transition_weight)
    
        expected_count, empirical_count = np.zeros((T, V)), np.zeros((T, V))
        for tag, word in itertools.product(range(T), range(V)):
            SumPotential = 0
        
            emission_score = np.exp(emission_weight[tag, word])
            if x[0] == word:
                transition_score = np.exp(transition_weight[-1, tag])
                SumPotential += backward_matrix[tag, 0]*emission_score*transition_score
            for i in range(1, N):
                if x[i] == word:
                    transition_scores = np.exp(transition_weight[:-1, tag])
                    SumPotential += np.sum(forward_matrix[:, i-1]*backward_matrix[tag, i]*emission_score*transition_scores)
            
            expected_count[tag, word] = SumPotential/NormalizationTerm
    
        Expected_count += expected_count

        for word, tag in zip(x, y):
            empirical_count[tag, word] += 1

        if counter % 100 == 0:
            print('Emission: done with the {}th instances'.format(counter))
        counter += 1

    L2_gradient = 2*param*emission_weight
    L2_gradient[L2_gradient == -np.inf] = 0

    return Expected_count - Empirical_count + L2_gradient

def GradientEmission(X, Y, tag2index, word2index, emission_weight, transition_weight, param=0.1):
    """
    The optimized version
    """
    T = len(tag2index)
    V = len(word2index)
    counter = 1
    Expected_count, Empirical_count = np.zeros((T, V)), np.zeros((T, V))
    
    for x, y in zip(X, Y):
        N = len(x)
        forward_matrix, NormalizationTerm = forward(x, tag2index, emission_weight, transition_weight)
        backward_matrix, _ = backward(x, tag2index, emission_weight, transition_weight)
    
        expected_count = np.zeros((T, V))


        emission_score = np.exp(emission_weight[:, x[0]]) # only apply to training set
        transition_score = np.exp(transition_weight[-1, :-1])
        expected_count[:, x[0]] += backward_matrix[:, 0]*emission_score*transition_score/NormalizationTerm
        
        for i in range(1, N):
            # for tag in range(T):
                # emission_score = np.exp(emission_weight[tag, x[i]])
                # transition_scores = np.exp(transition_weight[:-1, tag])
                # SumPotential += np.sum(forward_matrix[:, i-1]*backward_matrix[tag, i]*emission_score*transition_scores)
                # expected_count[tag, x[i]] = SumPotential/NormalizationTerm

            emission_score = np.exp(emission_weight[:, x[i]])
            transition_scores = np.exp(transition_weight[:-1, :-1])
            expected_count[:, x[i]] += np.sum(forward_matrix[:, i-1][:, None]*np.expand_dims(backward_matrix[:, i]*emission_score, axis=0)*transition_scores, axis=0)/NormalizationTerm
    
        Expected_count += expected_count

        for word, tag in zip(x, y):
            Empirical_count[tag, word] += 1

        if counter % 100 == 0:
            print('Emission: done with the {}th instances'.format(counter))
        counter += 1

    L2_gradient = 2*param*emission_weight
    L2_gradient[L2_gradient == -np.inf] = 0

    return Expected_count - Empirical_count + L2_gradient

def testTransition(epsilon, X, Y, tag1, tag2, tag2index, word2index, emission_weight, transition_weight, param=0.1):
    transition_weight_copy = copy.deepcopy(transition_weight)
    transition_weight_copy[tag1, tag2] += epsilon
    old_loss = Loss(X, Y, tag2index, emission_weight, transition_weight, param)
    new_loss = Loss(X, Y, tag2index, emission_weight, transition_weight_copy, param)
    gradient = GradientTransition(X, Y, tag2index, word2index, emission_weight, transition_weight, param)
    print('Actual loss change: {}, Change according to gradient: {}'.format(new_loss - old_loss, gradient[tag1, tag2]*epsilon))        

def testEmission(epsilon, X, Y, tag, word, tag2index, word2index, emission_weight, transition_weight, param=0.1):
    emission_weight_copy = copy.deepcopy(emission_weight)
    emission_weight_copy[tag, word] += epsilon
    old_loss = Loss(X, Y, tag2index, emission_weight, transition_weight, param)
    new_loss = Loss(X, Y, tag2index, emission_weight_copy, transition_weight, param)
    gradient = GradientEmission(X, Y, tag2index, word2index, emission_weight, transition_weight, param)
    print('Actual loss change: {}, Change according to gradient: {}'.format(new_loss - old_loss, gradient[tag, word]*epsilon))

def main():
    path = Path('../data/partial')
    train_X, train_X_str, train_Y, train_Y_str, test_X, test_X_str, test_Y, test_Y_str, word2index, tag2index, index2tag = get_xy(path)
    
    np.random.seed(1)
    T, V = len(tag2index), len(word2index)
    print('Number of tags: {}, Number of words: {}'.format(T, V))
    emission_weight = np.random.rand(T, V)
    transition_weight = np.random.rand(T+1, T+1)

    print('************Test*************')
    epsilon, tag1, tag2 = 0.00000001, 4, -1
    testTransition(epsilon, train_X, train_Y, tag1, tag2, tag2index, word2index, emission_weight, transition_weight, param=0.1)
    tag1, tag2 = -1, -1
    testTransition(epsilon, train_X, train_Y, tag1, tag2, tag2index, word2index, emission_weight, transition_weight, param=0.1)

    tag, word = 5, 5
    testEmission(epsilon, train_X, train_Y, tag, word, tag2index, word2index, emission_weight, transition_weight, param=0.1)

    def callbackF(w):
        loss = get_loss_grad(w)[0]
        print('Loss:{0:.4f}'.format(loss))

    def get_loss_grad(w):
        with HiddenPrints():
            transition_weight = w[:(T+1)*(T+1)].reshape((T+1, T+1))
            emission_weight = w[(T+1)*(T+1):].reshape((T, V))
            loss = Loss(train_X, train_Y, tag2index, 
                        emission_weight, transition_weight, param=0.1)
            grads_transition = GradientTransition(train_X[:5], train_Y[:5], tag2index, word2index, 
                                                emission_weight, transition_weight, param=0.1)
            grads_emission = GradientEmission(train_X[:5], train_Y[:5], tag2index, word2index, 
                                            emission_weight, transition_weight, param=0.1)
            grads = np.concatenate((grads_transition.reshape(-1), grads_emission.reshape(-1)))
        return loss, grads

    print('************Train*************')
    init_w = np.zeros(((T+1)*(T+1)+T*V,))
    optimal_weight, final_loss, result_dict = fmin_l_bfgs_b(get_loss_grad, init_w, pgtol=0.01, callback=callbackF)
    
    optimal_transition_weight = optimal_weight[:(T+1)*(T+1)].reshape((T+1, T+1))
    optimal_emission_weight = optimal_weight[(T+1)*(T+1):].reshape((T, V))
    path_transition = path/'best_weight_features1_transition.npy'
    path_emission = path/'best_weight_features1_emission.npy'
    np.save(path_transition, optimal_transition_weight)
    np.save(path_emission, optimal_emission_weight)

    path_output = path/'dev.p4.out'
    viterbi_output(path_output, test_X_str, test_X,  tag2index, optimal_emission_weight, optimal_transition_weight, link_weight_sum)
    
    prec, rec, f1 = eval(train_X, train_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, link_weight_sum)
    print('precision, recall, f1 on training set: {0} {1} {2}'.format(prec, rec, f1))
    prec, rec, f1 = eval(test_X, test_Y_str, tag2index, optimal_emission_weight, optimal_transition_weight, link_weight_sum)
    print('precision, recall, f1 on test set: {0} {1} {2}'.format(prec, rec, f1))


if __name__=='__main__':
    main()