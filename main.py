import cPickle
import gzip
import numpy as np
import scipy.stats as ss

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


def normalize(dot_result):
    #   min_val = np.min(dot_result)
    #   max_val = np.max(dot_result)
    #   return np.array([(x - min_val)/(max_val-min_val) for x in dot_result])
    arr_mean = np.mean(dot_result)
    arr_std = np.std(dot_result)
    return (dot_result - arr_mean)/arr_std
    #   return np.array(ss.zscore(input))


def max_activation(dot_result):
    max_val = np.max(dot_result)
    return np.where(dot_result == max_val)[0]


def train(trainer_set, nr_iter, learn_rate):
    #   weights_mat = np.zeros((10, 784), dtype=np.float)
    weights_mat = np.random.uniform(low=-3.0/784, high=3.0/784, size=(10, 784))
    bias_arr = np.array(10 * [0.0])

    success_rate = 0.0
    while success_rate < 89.0:
        all_number = len(trainer_set[0])

        for (pxl_arr, target) in zip(trainer_set[0], trainer_set[1]):

            z = normalize(np.dot(weights_mat, pxl_arr) + bias_arr)
            output_class = max_activation(z)
            true_output = z[target]
            error = 3.0 - true_output
            # if output_class != target:
            weights_mat[target] += pxl_arr * error * learn_rate
            bias_arr[target] += error * learn_rate
            weights_mat[output_class] -= pxl_arr * error * learn_rate
            bias_arr[output_class] -= error * learn_rate

            if output_class != target:
                all_number -= 1
        success_rate = (100.0 * all_number) / len(trainer_set[0])
        print success_rate
        nr_iter += 1
        print nr_iter
    return weights_mat, bias_arr


def test(tester_set, weights, bias_arr):
    well_classified = 0
    for (digit_arr, target) in zip(tester_set[0], tester_set[1]):

        z = normalize(np.dot(weights, digit_arr) + bias_arr)
        output_class = max_activation(z)
        if output_class == target:
            well_classified += 1

    return (100.0 * well_classified)/len(tester_set[0])

def multiple_perceptron_test(tester_set, weights, bias_arr):
    correct = 0
    for pxl_arr, target in zip(tester_set[0], tester_set[1]):
        results = np.array([ np.dot(pxl_arr, weight_arr) for weight_arr in weights ]) - np.array(bias_arr)
        if(max_activation(results) == target):
            correct+=1
    print correct
    return (100.0 * correct)/len(tester_set[0])

WEIGHTS, bias = train(train_set, 0, 0.05)
print WEIGHTS, bias

print test(test_set, WEIGHTS, bias)

with open('weights_file', 'w') as outfile:
    for row in WEIGHTS:
        for w in row:
            outfile.write(str(w) + " ")
        outfile.write("\n")

with open('bias_file_out', 'w') as outfile:
    for b in bias:
        outfile.write(str(b) + " ")
