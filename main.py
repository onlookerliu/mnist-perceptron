import cPickle, gzip, numpy as np, scipy.stats as ss, json
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def normalize(input):
    return (input - np.mean(input))/np.std(input)
    #return np.array(ss.zscore(input))

def max_activation(input):
    max = np.max(input)
    return np.where(input == max)[0]

def train(train_set, nr_iter, learn_rate):
    #PERCEPTRONS = np.zeros((10, 784), dtype=np.float)
    PERCEPTRONS = np.random.uniform(low=-3.0/784, high=3.0/784, size=(10, 784))
    bias = np.array(10 * [0.0])

    while(nr_iter > 0):
        all_number = len(train_set[0])
        for (pxl_arr, target) in zip(train_set[0], train_set[1]):

            z = normalize(np.dot(PERCEPTRONS, pxl_arr) + bias)
            output_class = max_activation(z)
            true_output = z[target]
            error = 3.0 - true_output
            if(output_class != target):
                PERCEPTRONS[target]+=pxl_arr*error*learn_rate
                bias[target] += error * learn_rate
                PERCEPTRONS[output_class] -= pxl_arr *error*learn_rate
                bias[output_class] -= error*learn_rate

            if(output_class != target):
                all_number -=1
        print all_number
        nr_iter -= 1
        print nr_iter
    return PERCEPTRONS, bias

def test(test_set, weights, bias):
    well_classified = 0
    for (digit_arr, target) in zip(test_set[0], test_set[1]):

        z = normalize(np.dot(weights, digit_arr) + bias)
        output_class = max_activation(z)
        if(output_class == target):
            well_classified +=1

    return (100.0 * well_classified)/len(test_set[0])


WEIGHTS, bias = train(train_set, 30, 0.08)
print WEIGHTS, bias

print test(test_set,WEIGHTS,bias)

with open('weights_file', 'w') as outfile:
    for row in WEIGHTS:
        for w in row:
            outfile.write(str(w) +" ")
        outfile.write("\n")

with open('bias_file_out', 'w') as outfile:
    for b in bias:
        outfile.write(str(b) +" ")