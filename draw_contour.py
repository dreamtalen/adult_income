import matplotlib.pyplot as plt
import pickle
import math

with open('search_result.pkl', 'rb') as f:
    cv_results_ = pickle.load(f)

# print cv_results_

train_set_size = 100

C_list = range(-5, 30)
gamma_list = range(-30, 5)

score_dict = [[[] for i in range(len(gamma_list))] for i in range(len(C_list))]

max_score = 0
best_c, best_gamma = 0, 0

c_value_list, gamma_value_list = [], []
for index, para in enumerate(cv_results_['params']):
    c = para['C']
    gamma = para['gamma']
    score = cv_results_['mean_test_score'][index]
    if score > max_score:
        max_score = score
        best_c = c
        best_gamma = gamma
    score_dict[C_list.index(int(math.log(c, 2)))][gamma_list.index(int(math.log(gamma, 2)))] =  score

print max_score
print best_c, best_gamma

# print score_dict
CS = plt.contour(gamma_list, C_list, score_dict)

plt.clabel(CS, inline=1, fontsize=10)

plt.xlabel('log(gamma)')
plt.ylabel('log(C)')
plt.savefig('grid_search_C_'+str(C_list[0])+'_'+str(C_list[-1])+'_gamma_'+str(gamma_list[0])+'_'+str(gamma_list[-1])+'_number'+str(train_set_size)+'.png')
plt.show()

