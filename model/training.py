
from controller_adjust import *

import sys

train_Report = train_for_some_generations(int(sys.argv[1]))

curr_solution = train_Report[1]

best_solution = train_Report[0][0]

print('=============== best_scores_in_generations_log')
print(len(best_scores_in_generations_log))
print()

best_solution_ever = copy.deepcopy(best_score_solution[1])

print('=============== best_solution_ever')
print(best_solution_ever)
print()

print('=============== best_solution')
print(best_solution)
print()

print('=============== curr_solution')
print(curr_solution)
print()

print('=============== best_score_solution')
print(best_score_solution)
print()

print('=============== worst_score_solution')
print(worst_score_solution)
print()


[Production_Error_in_dinamics, total_run_out_for_elecs, switch_nums_for_elecs,
            total_Production_in_dinamics, desired_total_Production_in_dinamics] = apply_solution(best_solution) #

[neg_J, RMSE, mean_error, asymetric_error, max_total_run_out, run_out_deviation_RMSE] = estimate_results([Production_Error_in_dinamics, total_run_out_for_elecs])

print('=========================== SCORE')
print('neg_J = ' + str(neg_J))
print('RMSE = ' + str(RMSE))
print('mean_error = ' + str(mean_error))
print('asymetric_error = ' + str(asymetric_error))
print('max_total_run_out = ' + str(max_total_run_out))
#print(min_max_tot_run_out)
print('run_out_deviation_RMSE = ' + str(run_out_deviation_RMSE))
print("switc num = ", end=' ')
print(switch_nums_for_elecs)
print(sum(switch_nums_for_elecs))
print()

plt.figure(figsize=(30, 15))
plt.title("Electrolyser modeling")
plt.plot(desired_total_Production_in_dinamics, label='I_ref')
plt.plot(total_Production_in_dinamics, label='I')

plt.legend()
plt.grid(visible=True)

num_elecs = 5
names = list(range(num_elecs))
values = total_run_out_for_elecs

plt.figure(figsize=(30, 15))
plt.bar(names, values)


plt.figure(figsize=(30, 15))
plt.title("log_generation")
plt.plot(best_scores_in_generations_log, label='I')

plt.legend()
plt.grid(visible=True)

plt.show()

import pickle

file_es = open('serialised_data_CMAES_MarkovDelta_t/evalution_strategy.pkl', 'wb')
pickle.dump(es.es, file_es)
file_es.close()


file_params = open('serialised_data_CMAES_MarkovDelta_t/best_params.pkl', 'wb')

file_log = open('serialised_data_CMAES_MarkovDelta_t/logs_relu_linear_norm_2out.pkl', 'wb')
file_dict_score_solution = open('serialised_data_CMAES_MarkovDelta_t/score_solution_relu_linear_norm_2out__number3.pkl', 'wb')

pickle.dump(best_solution, file_params)

pickle.dump(best_scores_in_generations_log, file_log)

max_min_score_solution = [best_score_solution, worst_score_solution]
pickle.dump(max_min_score_solution, file_dict_score_solution)

file_params.close()

file_log.close()
file_dict_score_solution.close()



