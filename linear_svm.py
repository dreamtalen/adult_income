import numpy as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_set_size')
parser.add_argument('test_set_size')
args = parser.parse_args()

train_set = np.loadtxt('adult.data.csv', delimiter=',')
test_set = np.loadtxt('adult.test.csv', delimiter=',')

train_set_size = int(args.train_set_size)
test_set_size = int(args.test_set_size)

X = train_set[:train_set_size, :-1]
y = train_set[:train_set_size, -1]

#scale the data attributes
X = preprocessing.scale(X)

# normalize the data attributes
# normalized_X = preprocessing.normalize(X)

# standardize the data attributes
# standardized_X = preprocessing.scale(X)


# print X[0]
# print scaled_X[0]
# print normalized_X[0]
# print standardized_X[0]

# X = scaled_X

test_X = test_set[:test_set_size, :-1]
test_y = test_set[:test_set_size, -1]

# normalized__test_X = preprocessing.normalize(test_X)
test_X = preprocessing.scale(test_X)


clf = LinearSVC()
clf.fit(X, y)

predicted = clf.predict(test_X)

wrong_num = sum(1 for i, j in zip(predicted, test_y) if i != j)

# print wrong_num
accurate = 1 - wrong_num/float(test_set_size)
with open('result.log', 'a') as f:
    f.write('LinearSVM'  + str(train_set_size) + '/' + str(test_set_size) + ' accurate: ' + str(accurate) + '\n')