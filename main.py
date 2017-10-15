import cPickle, gzip, numpy as np
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

def activation(input):
    max_val = np.max(input)
    print input
    return np.where(max_val==input)[0][0]


def normalize(input):
    s = sum(input)
    norm = np.array([float(i) / s for i in input])
    return norm


def train(train_set, nr_iter, train_rate):

    PERCEPTRONS = np.random.uniform(low=0.0, high=0.3, size=(784,10))
    all_classified = False

    while(not(all_classified) and nr_iter > 0 ):
        print nr_iter
        all_classified = True

        for (digit_arr, target) in zip(train_set[0], train_set[1]):
            dot_prod = np.dot(digit_arr, PERCEPTRONS)
            output = activation(dot_prod)
            print output

            PERCEPTRONS[:,target] = PERCEPTRONS[:,target]+(target-output)*digit_arr*train_rate



    return PERCEPTRONS



def test(test_set, weights):
    well_classified = 0
    for (digit_arr, target) in zip(test_set[0], test_set[1]):
        z = normalize(np.dot(weights, digit_arr) )

        if(target == activation(z)):
            well_classified +=1

    return (100.0 * well_classified)/len(test_set[0])


WEIGHTS = train(train_set, 2, 0.7)

print test(test_set, WEIGHTS)



