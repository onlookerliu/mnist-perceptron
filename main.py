import cPickle, gzip, numpy as np
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


# def normalize(input):
#     s = sum(input)
#     norm = np.array([float(i) / s for i in input])
#     return norm
#
# def activation_unbiased(dot_result, target):
#     max_val = np.max(dot_result)
#     if (max_val != 0.0):
#         if(np.where(dot_result == max_val) == target):
#             return target
#     return 0
#
#
# def train(train_set, nr_iter, train_rate):
#     PERCEPTRONS = np.zeros((10, 784), dtype=np.float)
#     all_classified = False
#
#     while(not(all_classified) and nr_iter > 0 ):
#         print nr_iter
#         all_classified = True
#         if(nr_iter < 6):
#             train_rate = 0.5
#         if(nr_iter < 3):
#             train_rate = 0.2
#
#         for (digit_arr, target) in zip(train_set[0], train_set[1]):
#             z = np.dot(PERCEPTRONS, digit_arr)
#             output = activation_unbiased(z, target)
#             PERCEPTRONS[target] = PERCEPTRONS[target] + (target-output)*digit_arr*train_rate
#             if(output != target):
#                 all_classified = False
#         nr_iter-=1
#
#     return PERCEPTRONS
#
#


def normalize(input):
    s = np.sum(input)
    return np.array([float(i)/s for i in input])

def activation(input, bias, target):
    return (input[target] - input[bias] > 0) * 1

def train(train_set, nr_iter, learn_rate):
    PERCEPTRONS = np.matrix((784,10), dtype = np.float)
    allClassified = False
    bias = np.array(10 * [0.0])

    while(allClassified == False and nr_iter > 0):
        for (pxl_arr, target) in zip(train_set[0], train_set[1]):

            z = np.dot(pxl_arr, PERCEPTRONS)
            output = activation(z, bias)
            PERCEPTRONS[:,target] += PERCEPTRONS[:,target] * (target-output)*learn_rate
            bias[target] += (target-output) * learn_rate
            if(output != target):
                allClassified = False
        nr_iter -= 1
        if(nr_iter < 20):
            learn_rate = 0.4
        if(nr_iter < 10):
            learn_rate = 0.2

def test(test_set, weights, bias):
    well_classified = 0
    for (digit_arr, target) in zip(test_set[0], test_set[1]):
        z = np.dot(weights, digit_arr)

        if(target == activation(z, bias, target)):
            well_classified +=1

    return (100.0 * well_classified)/len(test_set[0])


WEIGHTS = train(train_set, 20, 0.8)

