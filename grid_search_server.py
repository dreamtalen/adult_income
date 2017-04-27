import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import math
import pickle

train_set = np.loadtxt('adult.data.csv', delimiter=',')
test_set = np.loadtxt('adult.test.csv', delimiter=',')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('train_set_size')
args = parser.parse_args()

# Get the train_set size from command line parameters
train_set_size = int(args.train_set_size)
test_set_size = 1000

X = train_set[:train_set_size, :-1]
y = train_set[:train_set_size, -1]

# Scale the data attributes
X = preprocessing.scale(X)

test_X = test_set[:test_set_size, :-1]
test_y = test_set[:test_set_size, -1]

# Scale the data attributes
test_X = preprocessing.scale(test_X)

# Define the search range of parameters
C_list = range(-5, 30)
gamma_list = range(-30, 5)

parameters = {"C": [2**i for i in C_list], "gamma": [2**i for i in gamma_list]}
clf = GridSearchCV(svm.SVC(cache_size=1000), parameters, n_jobs=8)

clf.fit(X, y)

# Output the best parameters
print clf.best_params_
print math.log(clf.best_params_['C'], 2), math.log(clf.best_params_['gamma'], 2)
print clf.best_score_

# Output the grid search result to a file for contour drawing
with open('search_result'+str(C_list[0])+'_'+str(C_list[-1])+'_gamma_'+str(gamma_list[0])+'_'+str(gamma_list[-1])+'_number'+str(train_set_size)+'.pkl', 'wb') as f:
    pickle.dump(clf.cv_results_, f)

# print clf.cv_results_