import cPickle
import gzip
import numpy as np
from scipy.special import expit
import math

f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()


def max_activation(dot_result):
    max_val = np.max(dot_result)
    return np.where(dot_result == max_val)[0]


def mask_normalize(dot_result):
    min_val = abs(np.min(dot_result)) + 1.0
    masked_sum = len(dot_result) * min_val + np.sum(dot_result)
    return np.array([(float(x)+min_val)/masked_sum for x in dot_result])


def target_arr(target):
    target_result = np.array(10 * [0.0])
    target_result[target] = 1.0
    return target_result

def sigmoid(dot_result):
    return np.array([1.0/(1+math.exp(-x)) for x in dot_result])

def train(trainer_set, nr_iter, learn_rate):
    #   weights_mat = np.zeros((10, 784), dtype=np.float)
    weights_mat = np.random.uniform(low=-1.0/28, high=1.0/28, size=(10, 784))
    bias_arr = np.array(10 * [0.0])

    while nr_iter > 0:
        all_number = len(trainer_set[0])

        for (pxl_arr, target) in zip(trainer_set[0], trainer_set[1]):

            z = sigmoid(np.dot(weights_mat, pxl_arr) + bias_arr)
            output_class = max_activation(z)
            true_output = target_arr(target)
            if nr_iter == 1:
                print z, target
            if output_class != target:
                error_arr = true_output - z
                for idx, error in enumerate(error_arr):
                    weights_mat[idx] += pxl_arr * error * learn_rate
                    bias_arr[idx] += error * learn_rate

            if output_class != target:
                all_number -= 1
        success_rate = (100.0 * all_number) / len(trainer_set[0])
        print success_rate
        nr_iter -= 1
        print nr_iter
    return weights_mat, bias_arr


def test(tester_set, weights, bias_arr):
    well_classified = 0
    for (digit_arr, target) in zip(tester_set[0], tester_set[1]):

        z = sigmoid(np.dot(weights, digit_arr) + bias_arr)
        output_class = max_activation(z)
        if output_class == target:
            well_classified += 1

    return (100.0 * well_classified) / len(tester_set[0])


def multiple_perceptron_test(tester_set, weights, bias_arr):
    correct = 0
    for pxl_arr, target in zip(tester_set[0], tester_set[1]):
        results = np.array([np.dot(pxl_arr, weight_arr) for weight_arr in weights]) - np.array(bias_arr)
        if max_activation(results) == target:
            correct += 1
    print correct
    return (100.0 * correct) / len(tester_set[0])


WEIGHTS, bias = train(train_set, 40, 0.03)
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

