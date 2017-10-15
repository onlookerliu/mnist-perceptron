import cPickle, gzip, numpy as np
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


def activation_unbiased(dot_result, target):
    max_val = np.max(dot_result)
    if (max_val != 0.0):
        if(np.where(dot_result == max_val) == target):
            return target
    return 0


def activation(input):
    max_val = np.max(input)
    if( max_val >= 0.5):
       return np.where(input == max_val)[0]
    return 0

def normalize(input):
    s = sum(input)
    norm = np.array([float(i) / s for i in input])
    return norm

def arr_value(value):
    aux_arr = np.zeros(10, dtype=float)
    aux_arr[value] = 1.0
    return aux_arr

def train(train_set, nr_iter, train_rate):
    PERCEPTRONS = np.zeros((10, 784), dtype=np.float)
    all_classified = False
    bias = np.zeros(10, dtype=np.float)

    while(not(all_classified) and nr_iter > 0 ):
        print nr_iter
        all_classified = True
        if(nr_iter < 20):
            train_rate = 0.5
        if(nr_iter < 10):
            train_rate = 0.2

        for (digit_arr, target) in zip(train_set[0], train_set[1]):
            z = normalize(np.dot(PERCEPTRONS, digit_arr))+bias
            output = activation(z)
            PERCEPTRONS[target] = PERCEPTRONS[target] + (target-output)*digit_arr*train_rate
            bias = bias + (arr_value(target)-arr_value(output))*train_rate
            #change target function!
            if(output != target):
                all_classified = False
        nr_iter-=1

    return PERCEPTRONS, bias


def test(test_set, weights, bias):
    well_classified = 0
    for (digit_arr, target) in zip(test_set[0], test_set[1]):
        z = normalize(np.dot(weights, digit_arr) + bias)

        if(target == activation(z)):
            well_classified +=1

    return (100.0 * well_classified)/len(test_set[0])


WEIGHTS, bias = train(train_set, 1, 0.7)

print test(test_set, WEIGHTS, bias)

