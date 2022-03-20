
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from electrolyser import Electrolyser
from controller import Controller

import copy

import numpy as np

# [0, 1000, 5000, 3000, 300, 2400, 100, 0, 2500, 0, 1300, 200, 4000, 400, 4100, 5000, 500, 100, 0, 0, 1000, 1090,
    #  2000, 500]

# values = [0, 100, 500, 300, 300, 200, 100, 0, 200, 0, 100, 200, 400, 400, 400, 500, 500, 100, 0, 0, 100, 100, 200,
#           200]

values = [0, 100, 488, 488, 488, 320, 300, 200, 110, 265, 0, 488, 134, 200, 476, 490, 488, 500, 500, 100, 488,
                  70,  488, 80, 190, 280, 488, 488, 210]

# values = [0, 100]

#========================================================== PR_experimental = np.array(Irradiance_15min) / 2

from pandas import read_csv

Irradiance_data = read_csv('experimental_data/irradiance4.csv', delimiter=';')

Irradiance = Irradiance_data['Value'].to_numpy()[50:]

number_of_five_minutes = Irradiance.size

Irradiance_15min = []
Irradiance_1sec = []

delta_t = 1

t5 = 0

while t5 < number_of_five_minutes-3:
    quarter_of_hour = Irradiance[t5 : t5+3].mean()
    Irradiance_15min.append(quarter_of_hour)

    Irradiance_1sec.extend(quarter_of_hour * np.ones(int(15*60 / delta_t)))

    t5 += 3


PR_experimental = np.array(Irradiance_15min) / 2

#========================================================== PR_experimental = np.array(Irradiance_15min) / 2

def get_PRref_init():

    return np.random.choice(values, size=24) # 24 = 6 hours * (60 / 15) PR_experimental[:24] #

def norm_sample(mu, sigma):

    return norm(mu, sigma).rvs(size=1)[0]

def Generator_of_desired_PR(i, delta_t, Delta_t, PRref, PRref_noised) -> np.array:

    # к нему обращаются каждые Delta_t секунд (15 минут) и он возвращает временной ряда на следующие 6 часов
    # с разрешением Delta_t секунд, в котором первое значение, то есть на ближайшие секунд достоверное, а на последующие
    # 15минутки -
    # - это прогноз (зашумленные значениея). этот временной ряд каждые 15 минут обновляется (первое значение удаляется,
    # и в конец добавляется новое). Каждые 5 минут ряд поновому зашумляется, как буд то приходит новый прогноз.

    # обновление
    if i != 0:
        PRref = np.delete(PRref, 0)
        last = PRref[-1]
        next = last + norm_sample(mu = 0, sigma = 200)
        maxValue = 510
        minValue = 0
        next = max(min(next, maxValue), minValue)
        PRref = np.append(PRref, [next])

    # зашумление
    if i % int((Delta_t/3)/delta_t) == 0: # 5 минут
        PRref_noised[0] = PRref[0]
        for k in range(1, len(PRref)): # первое значение не зашумляется, оно достоверное
            alpha = PRref[k] * 5 / 100
            PRref_k = norm_sample(mu = PRref[k], sigma = alpha * k)
            extra = 0 # можно увеличить колокол, а то граничные значения имеют обрезанные колоколы
            PRref_noised[k] = max( min( PRref_k, max(values) + extra ), min(values) )

    # ==============

    # if i != 0:
    #     PRref = np.delete(PRref, 0)
    #     PRref = np.append(PRref, [PR_experimental[int(i * delta_t / Delta_t)]])
    #
    #     # зашумление
    # if i % int((Delta_t / 3) / delta_t) == 0:  # 5 минут
    #     PRref_noised[0] = PRref[0]
    #     for k in range(1, len(PRref)):  # первое значение не зашумляется, оно достоверное
    #         alpha = PRref[k] * 5 / 100
    #         PRref_k = PRref[k]  # norm_sample(mu = PRref[k], sigma = alpha * k)
    #         extra = 0  # можно увеличить колокол, а то граничные значения имеют обрезанные колоколы
    #         PRref_noised[k] = max(min(PRref_k, max(values) + extra), min(values))

    return [PRref, PRref_noised]

def init_Plant(num_of_elecs, init_states, delta_t):

    Plant = []
    IDs = []

    for j in range(num_of_elecs):
        elec = Electrolyser(j, delta_t)  # параметры по умолчанию
        [state, envTemperature, total_cost_of_work, total_run_out] = init_states[j]
        elec.set_init_state([state, envTemperature, total_cost_of_work, total_run_out])
        Plant.append(elec)
        IDs.append(j)

    return Plant

def simulate_PRref_generation(simulation_time, delta_t, Delta_t = 15*60):

    # simulate work process
    time_steps_in_simulation_time = int(simulation_time // delta_t)

    desired_total_Production_in_dinamics = []

    PRref = get_PRref_init()
    PRref_noised = copy.deepcopy(PRref)

    for i in range(time_steps_in_simulation_time):
        # каждые 15 минут генерируется кривая желаемой выработки incoming_curve_of_desired_production_rate на ближайшие
        # 6 часов с шагом 15 минут.

        if i % int(Delta_t/delta_t) == 0:

            [PRref, PRref_noised] = Generator_of_desired_PR(i, delta_t, Delta_t, PRref, PRref_noised)

            curve = []
            curve_noised = []

            for k in range(len(PRref)):
                value_at_next_15_minutes = PRref[k]
                value_at_next_15_minutes_noised = PRref_noised[k]
                curve.append(value_at_next_15_minutes * np.ones(int(15*60 / delta_t))) # 15 minutes
                curve_noised.append(value_at_next_15_minutes_noised * np.ones(int(15 * 60 / delta_t)))  # 15 minutes


            incoming_curve = np.concatenate(curve)
            incoming_curve_noised = np.concatenate(curve_noised)
            
            plt.figure(figsize=(30, 15))
            plt.title("PRref modeling")
            
            plt.plot(incoming_curve)
            plt.plot(incoming_curve_noised)

            plt.legend(['PR', 'PR_noised'])
            
            plt.show()

        desired_production_rate_in_moment = PRref_noised[0]  # истинное значение желаемой суммарной выработки в ближайшие 15 минут
        desired_total_Production_in_dinamics.append(desired_production_rate_in_moment)

    plt.figure(figsize=(30, 15))
    plt.title("PRref modeling")

    plt.plot(desired_total_Production_in_dinamics)

    # plt.legend(['PR', 'PR_noised'])

    plt.show()


def simulate_process(Plant, CTR, simulation_time, delta_t, Delta_t = 15*60, Generator_of_desired_PR = Generator_of_desired_PR):

    # simulate work process
    time_steps_in_simulation_time = int(simulation_time // delta_t)

    total_Production_in_dinamics = []
    desired_total_Production_in_dinamics = []  # тут записаны первые значение временного ряда (кривой желаемой выработки
    # на ближайшие несколько часов)
    Production_Error_in_dinamics = []

    PRref = get_PRref_init()
    PRref_noised = copy.deepcopy(PRref)
    
    Control_signals = []
    
    Output_derivative_for_elecs = []
    
    Output_dderivative_for_elecs = []
    
    for j in range(len(Plant)):
        Control_signals.append([])
        Output_derivative_for_elecs.append([])
        Output_dderivative_for_elecs.append([])

    U = []

    for i in range(time_steps_in_simulation_time):
        # каждые 15 минут генерируется кривая желаемой выработки incoming_curve_of_desired_production_rate на ближайшие
        # 6 часов с шагом 15 минут. этот ряд вместе с состояниями электролизеров поступают в контроллер.
        # он совершает действие, это действие отробатывается следующие 15 минут и так далее.

        if i % int(Delta_t / delta_t) == 0:
            [PRref, PRref_noised] = Generator_of_desired_PR(i, delta_t, Delta_t, PRref, PRref_noised)

            # для нее генерируется решение в данный момент врремени
            incoming_curve_of_desired_total_current = (53 / 100) * PRref_noised
            U = CTR.get_reference_ProductionRate_for_each_electrolyser(i, delta_t, incoming_curve_of_desired_total_current)

        # применяются решения ко всем электролизерам

        ######################################## STAT
        total_cost_of_work_for_elecs = []
        total_run_out_for_elecs = []
        States = []
        Targets = []
        Outs = []
        Temperatures = []

        switch_nums_for_elecs = []
        ######################################## STAT

        production_from_elecs_in_moment = 0

        for j in range(len(Plant)):
            elec = Plant[j]
            elec.apply_control_signal_in_moment(U[j], i * delta_t)

            [y, yd, ydd] = elec.getDinamics()
            [Temper, Temper_d] = elec.getTemperatureDinamics()

            production_from_elecs_in_moment += y * 100  # y это доля от максимального тока равного 53 ампера.
            # умножаем на 100 и получаем уровень выработки

            ######################################## STAT
            Outs.append(y * 53)
            Temperatures.append(Temper)

            total_cost_of_work_for_elecs.append(elec.total_cost_of_work)
            total_run_out_for_elecs.append(elec.total_run_out)
            States.append(elec.state)
            Targets.append(elec.getCurrentTarget())

            switch_nums_for_elecs.append(elec.switch_num)
            
            Control_signals[j].append(U[j])
            
            Output_derivative_for_elecs[j].append(yd * 53)
            Output_dderivative_for_elecs[j].append(ydd * 53)
            ######################################## STAT

        total_Production_in_dinamics.append(production_from_elecs_in_moment)

        desired_production_rate_in_moment = PRref_noised[0] # истинное значение желаемой суммарной выработки в ближайшие 15 минут
        desired_total_Production_in_dinamics.append(desired_production_rate_in_moment)

        Production_Error_in_dinamics.append((desired_production_rate_in_moment - production_from_elecs_in_moment))

    return [np.array(total_Production_in_dinamics),
            np.array(desired_total_Production_in_dinamics),
            np.array(Production_Error_in_dinamics),

            np.array(total_cost_of_work_for_elecs),
            np.array(total_run_out_for_elecs),

            np.array(Outs),
            np.array(Temperatures),
            np.array(States),
            np.array(Targets),
            np.array(switch_nums_for_elecs),
           
            np.array(Control_signals),
            
            np.array(Output_derivative_for_elecs),
            np.array(Output_dderivative_for_elecs)]


def get_agent_working_result(Agent, Generator_of_desired_PR = Generator_of_desired_PR):
    # spawn electrolyses
    num_of_elecs=5

    Plant = init_Plant(num_of_elecs=num_of_elecs, init_states=[['idle', 30.0, 0, 0] for _ in range(num_of_elecs)], delta_t=1)

    # init controller
    CTR = Controller()
    CTR.setPlant(Plant)

    CTR.setAgent(Agent)

    results = simulate_process(Plant, CTR, simulation_time = 60 * 60 * 6, delta_t = 1, Generator_of_desired_PR=Generator_of_desired_PR)

    return results


def estimate_results(results):

    [Production_Error_in_dinamics, total_run_out_for_elecs] = results

    RMSE = (np.dot(Production_Error_in_dinamics, Production_Error_in_dinamics) / Production_Error_in_dinamics.size)**0.5 # !

    mean_error = Production_Error_in_dinamics.mean()

    asymetric_error = asymetric_plus(mean_error/10, 2, 1) # !

    max_total_run_out = total_run_out_for_elecs.max()
    mean_total_run_out = total_run_out_for_elecs.mean()
    min_total_run_out = total_run_out_for_elecs.min()

    #min_max_tot_run_out = (max_total_run_out - min_total_run_out) / min_total_run_out # !

    deviation = total_run_out_for_elecs - ( mean_total_run_out * np.ones_like(total_run_out_for_elecs) )

    run_out_deviation_RMSE = ( np.dot(deviation, deviation) / deviation.size )**0.5 # !

    gamma1 = 0.3
    gamma2 = 0.2
    gamma3 = 0.00002 # 0.00002
    gamma4 = 0.00002
    gamma5 = 0.00002

    #J = np.log(alpha * RMSE) #+ np.log(betha * max_tatal_run_out)

    J = gamma1 * RMSE + gamma2 * asymetric_error + gamma3 * max_total_run_out + gamma5 * run_out_deviation_RMSE # gamma4 * min_max_tot_run_out

    return [-J, RMSE, mean_error, asymetric_error, max_total_run_out, run_out_deviation_RMSE]


def asymetric_plus(x, a, b):

    return (np.log(1 + np.e**(x)) * a)**2 + (np.log(1 + np.e**(-x)) / b)**2

def account_reward():
    pass


# from network_arch import Agent
#
# if __name__ == '__main__':
#
#     agent = Agent()
#
#     results = get_agent_working_result(agent)
#
#     [total_Production_in_dinamics,
#      desired_total_Production_in_dinamics,
#      Production_Error_in_dinamics,
#
#      total_cost_of_work_for_elecs,
#      total_run_out_for_elecs,
#
#      Outs,
#      Temperatures,
#      States,
#      Targets]                                       = results
#
#     ######################################## STAT
#     print('total_cost_of_work_for_elecs', end=' = ')
#     print(total_cost_of_work_for_elecs)
#     print('total_run_out_for_elecs', end=' = ')
#     print(total_run_out_for_elecs)
#     print("State = " + str(States))
#     print("Targets = " + str(Targets))
#
#     print("Outs = " + str(Outs))
#     print("Temperatures = " + str(Temperatures))
#     print()
#     ######################################## STAT
#
#     RMSE = (sum(Production_Error_in_dinamics) / len(Production_Error_in_dinamics)) ** 0.5
#     total_Production = sum(total_Production_in_dinamics)
#     desired_total_Production = sum(desired_total_Production_in_dinamics)
#     print("RMSE_Production_Rate = " + str(RMSE))
#     print("RMSE_Production_Rate / total_Production = " + str(
#         RMSE / (total_Production)))  # сколько потеряли по отношению к тому сколько получили
#     print("RMSE_Production_Rate / desired_total_Production = " + str(
#         RMSE / desired_total_Production))  # сколько потеряли по отношению к тому сколько хотели получить
#     print("(desired_total_Production - total_Production)/desired_total_Production) = " + str(
#         (desired_total_Production - total_Production) / desired_total_Production))
#
