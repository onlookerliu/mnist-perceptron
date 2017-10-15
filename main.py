import cPickle, gzip, numpy as np
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def activation(input):
    max_val = np.max(input)
    print input
    return np.where(max_val==input)[0]


def normalize(input):
    s = sum(input)
    max_val = max(input)
    min_val = min(input)
    norm = np.array([(i-min_val) / (max_val-min_val) for i in input])
    return norm


"""def train(train_set, nr_iter, train_rate):
    #PERCEPTRONS = np.zeros((10, 784), dtype=np.float)
    PERCEPTRONS = np.random.uniform(low=0.0, high=0.3, size=(10,784))
    all_classified = False

    while(not(all_classified) and nr_iter > 0 ):
        print nr_iter
        all_classified = True
        if(nr_iter < 20):
            train_rate = 0.5
        if(nr_iter < 10):
            train_rate = 0.2

        for (digit_arr, target) in zip(train_set[0], train_set[1]):
            dot_prod = np.dot(PERCEPTRONS, digit_arr)
            output = activation(dot_prod)
            print target
            first_tup = (target-output)*digit_arr*train_rate
            PERCEPTRONS[target] = PERCEPTRONS[target] + first_tup
            #change target function!
            if(output != target):
                all_classified = False
        nr_iter-=1

    return PERCEPTRONS
"""

def train(train_set, nr_iter, train_rate):
    PERCEPTRONS = np.random.uniform(low=0.0, high = 0.3, size=(784,10))
    all_classified = False

    while(all_classified == False):
        all_classified = True
        good_class = 0
        for(digit_arr, target) in zip(train_set[0], train_set[1]):
            pxl_mat = np.mat(digit_arr)
            dot_prod = np.dot(digit_arr, PERCEPTRONS)
            output = activation(dot_prod)
            PERCEPTRONS[:, target] = PERCEPTRONS[:, target]+(target-output)*pxl_mat*train_rate
            if output!=target:
                all_classified=False
            else:
                good_class+=1
        print good_class


def test(test_set, weights):
    well_classified = 0
    for (digit_arr, target) in zip(test_set[0], test_set[1]):
        z = normalize(np.dot(weights, digit_arr) )

        if(target == activation(z)):
            well_classified +=1

    return (100.0 * well_classified)/len(test_set[0])


WEIGHTS = train(train_set, 2, 0.7)

print test(test_set, WEIGHTS)



