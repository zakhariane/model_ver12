
import numpy as np
import matplotlib.pyplot as plt
import cma
from es import *
from electrolyser import Electrolyser
from network_arch import Agent
from simulation import get_agent_working_result, estimate_results

import copy

agent = Agent()


def apply_solution(solution):

    agent.updateParams(solution)

    [total_Production_in_dinamics,
     desired_total_Production_in_dinamics,
     Production_Error_in_dinamics,

     total_cost_of_work_for_elecs,
     total_run_out_for_elecs,

     Outs,
     Temperatures,
     States,
     Targets,
     switch_nums_for_elecs,
     Control_signals,

     Output_derivative_for_elecs,
     Output_dderivative_for_elecs]                                       = get_agent_working_result(agent)

    return [Production_Error_in_dinamics, total_run_out_for_elecs, switch_nums_for_elecs,
            total_Production_in_dinamics, desired_total_Production_in_dinamics]


param_count = 0
for param in agent.model.parameters():
  print(param.data.shape)
  param_count += np.product(param.data.shape)
print(param_count)


import pickle

file_es = open('serialised_data_CMAES_MarkovDelta_t/evalution_strategy.pkl', 'rb')
old_es = pickle.load(file_es)
file_es.close()

sigma_init = 0.5 # 0.1

popsize = 100

es = CMAES(num_params=param_count, sigma_init = sigma_init, popsize=popsize)

es.es = old_es

del old_es


file_params = open('serialised_data_CMAES_MarkovDelta_t/best_params.pkl', 'rb')

file_log = open('serialised_data_CMAES_MarkovDelta_t/logs_relu_linear_norm_2out.pkl', 'rb')
file_dict_score_solution = open('serialised_data_CMAES_MarkovDelta_t/score_solution_relu_linear_norm_2out__number3.pkl', 'rb')


best_params = pickle.load(file_params)

best_scores_in_generations_log = pickle.load(file_log)
max_min_score_solution = pickle.load(file_dict_score_solution)


file_params.close()

file_log.close()
file_dict_score_solution.close()


best_score_solution = max_min_score_solution[0]
worst_score_solution = max_min_score_solution[1]


def train_for_some_generations(generations_num):
    for i in range(generations_num):

        solutions = es.ask()

        reward_list = []

        solut_number = 0
        for solution in solutions:  # можно параллельно

            [Production_Error_in_dinamics, total_run_out_for_elecs, switch_nums_for_elecs,
             total_Production_in_dinamics, desired_total_Production_in_dinamics] = apply_solution(solution)
            score = estimate_results([Production_Error_in_dinamics, total_run_out_for_elecs])[
                0]  # get cost functions value at point solution

            reward_list.append(score)

            print(str(solut_number) + ' solution is aplied, score = ' + str(score) + '  min-max = ' + str(
                min(solution)) + ' -- ' + str(max(solution)))

            if score > best_score_solution[0]:
                best_score_solution[0] = copy.deepcopy(score)
                best_score_solution[1] = copy.deepcopy(solution)

            if score < worst_score_solution[0]:
                worst_score_solution[0] = copy.deepcopy(score)
                worst_score_solution[1] = copy.deepcopy(solution)

            solut_number += 1

        es.tell(reward_list)

        es_solution = es.result()

        #         model_params = es_solution[0] # best historical solution

        best_reward = es_solution[1]  # best reward
        curr_best_reward = es_solution[2]  # best of the current batch

        curr_best_reward_my_for_validation = max(reward_list)  # min with '-'

        print(str(i) + "  ==>>", end=' ')
        print(curr_best_reward_my_for_validation, end=' === ')
        print(curr_best_reward, end=' === ')
        print(best_reward, end=' === ')
        print(es.rms_stdev())

        best_scores_in_generations_log.append(curr_best_reward_my_for_validation)

    return [es.result(), es.current_param()]  # best historical solution

