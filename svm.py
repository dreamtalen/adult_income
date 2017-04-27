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

# Get the train_set and test_set size from command line parameters
train_set_size = int(args.train_set_size)
test_set_size = int(args.test_set_size)

X = train_set[:train_set_size, :-1]
y = train_set[:train_set_size, -1]

# Scale the data attributes
X = preprocessing.scale(X)

test_X = test_set[:test_set_size, :-1]
test_y = test_set[:test_set_size, -1]

# Scale the data attributes
test_X = preprocessing.scale(test_X)

# Set the parameter due to the grid search result
c = 2**7
gamma = 2**-7
clf = SVC(kernel=args.argument, cache_size=4000, C=c, gamma=gamma)

clf.fit(X, y)

predicted = clf.predict(test_X)

# Calculate the accuracy
wrong_num = sum(1 for i, j in zip(predicted, test_y) if i != j)
accurate = 1 - wrong_num/float(test_set_size)

# Output the experiment result to log file
with open('result.log', 'a') as f:
    f.write(args.argument + ' '+ str(train_set_size) + '/' + str(test_set_size) + ' accurate: ' + str(accurate) + ' c=' + str(c) +' gamma=' + str(gamma) + '\n')