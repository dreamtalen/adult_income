import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_set_size')
parser.add_argument('test_set_size')
parser.add_argument('argument')
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
# X = preprocessing.normalize(X)


test_X = test_set[:test_set_size, :-1]

test_y = test_set[:test_set_size, -1]

#scale the data attributes
test_X = preprocessing.scale(test_X)

# normalize the data attributes
# test_X = preprocessing.normalize(test_X)

c = 32
gamma = 0.0078125
clf = SVC(kernel=args.argument, cache_size=4000, C=c, gamma=gamma) #86%
# print clf
# clf.fit(X, y)
clf.fit(X, y)

predicted = clf.predict(test_X)

wrong_num = sum(1 for i, j in zip(predicted, test_y) if i != j)

# print wrong_num
accurate = 1 - wrong_num/float(test_set_size)
with open('result.log', 'a') as f:
    f.write(args.argument + ' '+ str(train_set_size) + '/' + str(test_set_size) + ' accurate: ' + str(accurate) + ' c=' + str(c) +' gamma=' + str(gamma) + '\n')