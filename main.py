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

def normalize(input):
    s = sum(input)
    norm = np.array([float(i) / s for i in input])
    return norm

def train(train_set, nr_iter, train_rate):
    PERCEPTRONS = np.zeros((10, 784), dtype=np.float)
    all_classified = False

    while(not(all_classified) and nr_iter > 0 ):
        print nr_iter
        all_classified = True
        if(nr_iter < 6):
            train_rate = 0.5
        if(nr_iter < 3):
            train_rate = 0.2

        for (digit_arr, target) in zip(train_set[0], train_set[1]):
            z = np.dot(PERCEPTRONS, digit_arr)
            output = activation_unbiased(z, target)
            PERCEPTRONS[target] = PERCEPTRONS[target] + (target-output)*digit_arr*train_rate
            if(output != target):
                all_classified = False
        nr_iter-=1

    return PERCEPTRONS


def test(test_set, weights):
    well_classified = 0
    for (digit_arr, target) in zip(test_set[0], test_set[1]):
        z = np.dot(weights, digit_arr)

        if(target == activation_unbiased(z, target)):
            well_classified +=1

    return (100.0 * well_classified)/len(test_set[0])


WEIGHTS = train(train_set, 10, 0.6)

print test(test_set, WEIGHTS)


