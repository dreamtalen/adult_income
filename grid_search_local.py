import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import pandas as pd
import math
import matplotlib.pyplot as plt
import time
train_set = np.loadtxt('adult.data.csv', delimiter=',')
test_set = np.loadtxt('adult.test.csv', delimiter=',')

train_set_size = 500
test_set_size = 1000

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
# C_list = [i/2.0 for i in range(0, 12)]
# gamma_list = [i/2.0 for i in range(-9, -3)]
C_list = range(-5, 40)
gamma_list = range(-40, 5)

parameters = {"C": [2**i for i in C_list], "gamma": [2**i for i in gamma_list]}
clf = GridSearchCV(svm.SVC(cache_size=1000), parameters, n_jobs=-1)
# print clf
# clf.fit(X, y)
clf.fit(X, y)

# print clf.best_estimator_
# print pd.DataFrame(clf.cv_results_)
print clf.best_params_
print math.log(clf.best_params_['C'], 2), math.log(clf.best_params_['gamma'], 2)
print clf.best_score_

score_dict = [[[] for i in range(len(gamma_list))] for i in range(len(C_list))]

c_value_list, gamma_value_list = [], []
for index, para in enumerate(clf.cv_results_['params']):
    c = para['C']
    gamma = para['gamma']
    score = clf.cv_results_['mean_test_score'][index]
    # if c not in c_value_list:   c_value_list.append(c)
    # if gamma not in gamma_value_list:   gamma_value_list.append(gamma)
    # score_dict[c_value_list.index(c)][gamma_value_list.index(gamma)] = score
    score_dict[C_list.index(int(math.log(c, 2)))][gamma_list.index(int(math.log(gamma, 2)))] =  score

# CS = plt.contour(gamma_value_list, c_value_list, score_dict)
CS = plt.contour(gamma_list, C_list, score_dict)
# manual_locations = [(int(math.log(clf.best_params_['gamma'], 2)), int(math.log(clf.best_params_['C'], 2)))]
plt.clabel(CS, inline=1, fontsize=10)
# plt.semilogx(gamma_value_list, c_value_list)
plt.xlabel('log(gamma)')
plt.ylabel('log(C)')
plt.savefig('grid_search'+str(C_list[0])+'_'+str(C_list[-1])+'.png')
plt.show()
# predicted = clf.predict(test_X)
# time.sleep(100)
# wrong_num = sum(1 for i, j in zip(predicted, test_y) if i != j)

# accurate = 1 - wrong_num/float(test_set_size)