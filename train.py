import sys
import math

def print_graph(test_id):
    start = test_id * 28
    end = start + 28
    file = open('testimages')
    for i, line in enumerate(file):
        if i >= start and i < end:
            print line
    file.close()

def parse_features(fname, N):
    arr = [[[0 for k in xrange(28)] for j in xrange(28)] for i in xrange(N)]   # num cols, num rows, num examples
    file = open(fname)
    for i in range(0,N):
        for y in range(0,28):
            for x in range(0,29):    # need to consume the '\n' char
                char = file.read(1)
                if char == '+' or char == '#':
                    arr[i][y][x] = 1
    
    file.close()
    return arr

def parse_labels(fname, N):
    arr = []
    file = open(fname)
    for line in file:
        arr.append(int(line))
    file.close()
    return arr

def print_digit(arr):
    for i in arr:
        print i

# computes likelihood of 1 in class digit at position (x, y)
def likelihood(x, y, features, labels, N, digit, k, f): 
    count = 0
    freq = 0
    for i in range(0,N):
        if labels[i] == digit:
            freq += 1
            if features[i][y][x] == f:
                count += 1
    return (count+k)/(freq+2.0*k)          # laplace smoothing

def prior(digit, labels, N):
    count = 0
    for i in range(0,N):
        if labels[i] == digit:
            count += 1
    return count/(N+0.0)

def map_decision(priors, posteriors, test, N):
    maps = []
    for digit in range(0,10):
        val = math.log(priors[digit])
        for y in range(0,28):
            for x in range(0,28):
                f = test[y][x]
                val += math.log(posteriors[digit][y][x][f])

        maps.append(val)
    
    return maps.index(max(maps))  

    

if __name__ == '__main__':
    # arrays contraining train and test
    train = parse_features('trainingimages', 5000)
    train_labels = parse_labels('traininglabels', 5000)
    test = parse_features('testimages', 1000)
    test_labels = parse_labels('testlabels', 1000)
    smoothing_constant = 1

    # build learned model
    priors = []
    posteriors = [[[[0 for i in xrange(2)] for k in xrange(28)] for j in xrange(28)] for z in xrange(10)] # digit,y,x,f
    for i in range(0,10):
        priors.append(prior(i, train_labels, 5000))
        for y in range(0,28):
            for x in range(0,28):
                for f in range(0,2):
                    posteriors[i][y][x][f] = likelihood(x,y,train,train_labels,5000,i,smoothing_constant,f) 


    # decision making
    keys = range(10)
    max_posteriors = {key: (0,0) for key in keys} # dict of digit as key, val as pair of indix of the tests and posterior
    min_posteriors = {key: (0,0) for key in keys}
    for i in range(0,1000):
        for digit in range(0,10):
            posterior_sum = 0.0
            for y in range(0,28):
                for x in range(0,28):
                    f = test[i][y][x]
                    posterior_sum += posteriors[digit][y][x][f]
            if max_posteriors[digit][1] < posterior_sum:   # update max
                max_posteriors[digit] = (i, posterior_sum)
            if min_posteriors[digit][1] == 0:   # initialize
                min_posteriors[digit] = (i, posterior_sum)
            if min_posteriors[digit][1] > posterior_sum:   # update min
                min_posteriors[digit] = (i, posterior_sum)

    print max_posteriors
    for i in max_posteriors:
        print i
        print_graph(max_posteriors[i][0])
    print min_posteriors
    for i in min_posteriors:
        print i
        print_graph(max_posteriors[i][0])

"""
    # version 1, code piece for finding smoothing constant
    accuracy = 0
    for i in range(0,1000):
        guess = map_decision(priors, posteriors, test[i], 1000)
        if guess == test_labels[i]:
            accuracy += 1
    accuracy /= 1000.0
    print accuracy


    # version 2, code piece for reporting classification rates for digits
    rates = [0.0 for i in range(10)]
    totals = [0 for i in range(10)]
    for i in range(0,1000):
        digit = test_labels[i]
        totals[digit] += 1
        guess = map_decision(priors, posteriors, test[i], 1000)
        if guess == digit:
            rates[digit] += 1
    for i in range(0,10):
        print rates[i] / totals[i]


    # version 3, code piece for confusion matrix
    confusion_counts = [[0 for i in range(10)] for j in range(10)]
    confusion_totals = [[0 for i in range(10)] for j in range(10)]
    confusion = [[0 for i in range(10)] for j in range(10)] 
    for i in range(0,1000):
        digit = test_labels[i]
        for j in range(0,10):
            confusion_totals[digit][j] += 1    # the whole digit row +=1
        guess = map_decision(priors, posteriors, test[i], 1000)
        confusion_counts[digit][guess] += 1
    
    for i in range(0,10):
        for j in range(0,10):
            rate = (confusion_counts[i][j]+0.0) / confusion_totals[i][j]
            confusion[i][j] = "{0:.3f}".format(rate)
    for i in range(0,10):
        print confusion[i]

"""
