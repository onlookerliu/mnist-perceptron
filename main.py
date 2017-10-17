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

def activation(input, target):
    return (input[target] > 0.5) * 1

def train(train_set, nr_iter, learn_rate):
    PERCEPTRONS = np.random.rand(784, 10)
    allClassified = False
    bias = np.array(10 * [0.0])

    while(allClassified == False and nr_iter > 0):
        for (pxl_arr, target) in zip(train_set[0], train_set[1]):

            z = normalize(np.dot(pxl_arr, PERCEPTRONS))
            output = activation(z, target)
            PERCEPTRONS[:,target] += pxl_arr * (1.0-z[target])*learn_rate
            if(output != 1):
                bias[target] += (1.0 - z[target]) * learn_rate
                allClassified = False
        nr_iter -= 1
        if(nr_iter < 20):
            learn_rate = 0.03
        if(nr_iter < 10):
            learn_rate = 0.01
        print nr_iter
    return PERCEPTRONS, bias

def test(test_set, weights, bias):
    well_classified = 0
    for (digit_arr, target) in zip(test_set[0], test_set[1]):
        z = np.dot(weights, digit_arr)

        if(target == activation(z, bias, target)):
            well_classified +=1

    return (100.0 * well_classified)/len(test_set[0])


WEIGHTS, bias = train(train_set, 40, 0.05)
print WEIGHTS, bias

