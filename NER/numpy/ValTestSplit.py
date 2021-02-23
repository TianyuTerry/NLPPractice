from pathlib import Path

def val_test_split(path, val_ratio, path_val, path_test):
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

    split_point = int(len(X)*val_ratio)
    val_X, val_Y = X[:split_point], Y[:split_point]
    test_X, test_Y = X[split_point:], Y[split_point:]

    dump(path_val, val_X, val_Y)
    dump(path_test, test_X, test_Y)

def val_test_split_pos(path, val_ratio, path_val, path_test):
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

    split_point = int(len(X)*val_ratio)
    val_X, val_Y = X[:split_point], Y[:split_point]
    test_X, test_Y = X[split_point:], Y[split_point:]

    dump_pos(path_val, val_X, val_Y)
    dump_pos(path_test, test_X, test_Y)


def dump(path, X, Y):
    output_string = ''
    for i in range(len(X)):
        for j in range(len(X[i])):
            output_string += X[i][j] + ' ' + Y[i][j] + '\n'
        output_string += '\n'
    
    with open(path, 'w') as f:
        f.write(output_string)
    
    print('Dumping done')
    return None

def dump_pos(path, X, Y):
    output_string = ''
    for i in range(len(X)):
        for j in range(len(X[i])):
            output_string += X[i][j][0] + ' ' + X[i][j][1] + ' ' + Y[i][j] + '\n'
        output_string += '\n'
    
    with open(path, 'w') as f:
        f.write(output_string)
    
    print('Dumping done')
    return None

def main():
    root = Path('../data/partial')
    path, path_val, path_test = root/'dev.out', root/'dev', root/'test'
    val_test_split(path, 0.5, path_val, path_test)
    root = Path('../data/full')
    path, path_val, path_test = root/'dev.out', root/'dev', root/'test'
    val_test_split_pos(path, 0.5, path_val, path_test)

if __name__=='__main__':
    main()