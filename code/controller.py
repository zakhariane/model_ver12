
import numpy as np
from electrolyser import Electrolyser
from network_arch import Agent

class Controller:
    def __init__(self):
        self.Plant = []
        self.Agent = Agent()
        self.experimental_control_signal = np.array([]) 

    def setPlant(self, Plant):
        self.Plant = Plant

    def setAgent(self, Agent):
        self.Agent = Agent

    # def update_params_agent(self, params):
    #     self.Agent.updateParams(params)
    

    def get_reference_ProductionRate_for_each_electrolyser(self, i, delta_t, curve_of_desired_total_current: np.array):
        # curve_of_desired_total_current[i] is a reference total CURRENT at time t + i*delta_t

        #U = self.exampleController(delta_t, curve_of_desired_total_current)
        U = self.RL_controller(i, delta_t, curve_of_desired_total_current)
        
        #U = self.generate_experimental_control_for_identification(i, delta_t)

        return U # U[i] \in {0%} \cup [60%,100%]
    
    
    def generate_experimental_control_for_identification(self, i, delta_t):
        
        n = len(self.Plant)
        
        U = []
        
        for j in range(n):
            U.append(self.experimental_control_signal[i])
            
        U = np.array(U)
        
        return U


    def exampleController(self, delta_t, curve_of_desired_total_current: np.array):
        # curve_of_desired_total_current[i] is a reference total CURRENT at time t + i*delta_t

        #########
        maxOutput_of_electrolyser = 53
        #########

        # hour = 60*60
        # N_hour = int(hour // delta_t)
        #
        # desired_Rate_at_next_hour = np.average(curve_of_desired_total_current[:N_hour])

        desired_Rate_at_next_hour = curve_of_desired_total_current[0]

        Total_output = 0
        for i in range(len(self.Plant)):
            Total_output += self.Plant[i].getCurrentTarget()*maxOutput_of_electrolyser # target (set point) current of electrolyser number i

        working_elecs = []
        not_working_elecs = []
        U = [0] * len(self.Plant)
        for i in range(len(self.Plant)):
            elec = self.Plant[i]
            if elec.getCurrentTarget() != 0:
                working_elecs.append(elec)
            else:
                not_working_elecs.append(elec)
            U[i] = elec.getCurrentTarget() * 100

        number_of_required_new_electrolysers = 0
        newTotal_output = Total_output
        while abs(newTotal_output - desired_Rate_at_next_hour) > maxOutput_of_electrolyser/2: # ошибаемся больше чем на половину выработки одного электролизера
            if newTotal_output < desired_Rate_at_next_hour:
                number_of_required_new_electrolysers += 1
                newTotal_output += maxOutput_of_electrolyser
            else: # newTotal_output > desired_Rate_at_next_hour
                number_of_required_new_electrolysers -= 1
                newTotal_output -= maxOutput_of_electrolyser

        if number_of_required_new_electrolysers < 0: # нужно выключить несколько
            # выключаем те которые самые горячие
            # выключаем те которые самые изношенные *
            # выключаем те которые дольше всех работают

            working_elecs = sorted(working_elecs, key = lambda x: -x.total_run_out )
            num_of_elecs_to_off = -number_of_required_new_electrolysers
            for i in range(min(num_of_elecs_to_off, len(working_elecs))):
                U[working_elecs[i].getID()] = 0

        elif number_of_required_new_electrolysers > 0: # нужно включить несколько
            # включаем те которые дольше всех выключены
            # самые холодные *
            # самые не изношенные

            not_working_elecs = sorted(not_working_elecs, key=lambda x: x.getTemperatureDinamics()[0])
            for i in range(min(number_of_required_new_electrolysers, len(not_working_elecs))):
                U[not_working_elecs[i].getID()] = 100

        return U # U[i] \in {0%} \cup [60%,100%]

    def RL_controller(self, i, delta_t, curve_of_desired_total_current: np.array):
        # curve_of_desired_total_current[i] is a reference total CURRENT at time t + i*delta_t

        def formate_state(Electrolysers, desired_curve):
            # desired_curve это кривая желаемого значения тока со всех электролизеров на ближайшие несколько часов

            # получние максимальных значений для нормализации
            Imax = 53 + 0.2
            Tmax = Electrolysers[0].maxTemperature + 0.2
            max_value_desired_curve = len(Electrolysers) * Imax
            max_value_El_Targets = 1.0
            max_value_El_Currents = 1 # из elec.getDinamics() поступают уже нормированные значения
            max_value_El_Temperatures = Tmax

            x_desired_curve = desired_curve.copy() / max_value_desired_curve # 6 часов с разрешением 15 минут => 24 значения в массиве

            El_Targets = []
            El_Currents = []
            El_Currents_dot = []
            El_Temperatures = []
            El_States = []
            El_RunOuts = []
            # El_Consumptions = [] # не играет роли пока нет конкретных значений потребления в том или ином режиме. пока оно просто повторяет RunOuts

            for j in range(len(Electrolysers)):
                elec = Electrolysers[j]

                El_Targets.append(elec.getCurrentTarget()) # max value = 1
                El_Currents.append(elec.getDinamics()[0]) # max value = 1
                El_Currents_dot.append(elec.getDinamics()[1]) # max value = 1
                El_Temperatures.append(elec.getTemperatureDinamics()[0]) # max value = Tmax

                El_States.append(elec.states_list.index(elec.getState())) # max value = 1 (это one hot вектор)

                El_RunOuts.append(elec.getRunOut()) # max value = сумма total_run_out по всем электролизерам

            x_El_Targets = np.array(El_Targets) / max_value_El_Targets # max value = 1
            x_El_Currents = np.array(El_Currents) / max_value_El_Currents # max value = 1
            x_El_Currents_dot = np.array(El_Currents_dot) / max_value_El_Currents  # max value = 1
            x_El_Temperatures = np.array(El_Temperatures) / max_value_El_Temperatures # max value = Tmax

            x_El_States = np.array(El_States)

            x_El_States_one_hot = np.zeros((x_El_States.size, len(Electrolysers[0].states_list)))
            x_El_States_one_hot[np.arange(x_El_States.size), x_El_States] = 1

            x_El_States_one_hot = np.reshape(x_El_States_one_hot, (1, len(Electrolysers[0].states_list) * len(Electrolysers)))[0]

            # if i % int(10 * 60 / 1) == 0:
            #     ST = []
            #     for j in range(len(Electrolysers)):
            #         onehot_state_j = list(x_3np1_4n_one_hot[j*10 : (j+1)*10 ])
            #         indx = onehot_state_j.index(1)
            #
            #         S = ['idle', 'heating', 'hydration', 'ramp_up_1', 'ramp_up_2', 'steady',
            #          'ramp_down_1', 'ramp_down_2', 'offline', 'error'] [indx]
            #
            #         ST.append(S)
            #
            #     print(ST)

            x_El_RunOuts = np.array(El_RunOuts) # max value = сумма total_run_out по всем электролизерам
            sum_runout = sum(El_RunOuts)
            if sum_runout != 0:
                x_El_RunOuts /= sum_runout

            state = np.concatenate((x_desired_curve, x_El_Targets, x_El_Currents, x_El_Currents_dot, x_El_Temperatures,
                                    x_El_States_one_hot, x_El_RunOuts))

            return state

        state = formate_state(self.Plant, curve_of_desired_total_current)

        n = len(self.Plant)

        U_logist_regression = self.Agent.getAction(state, i, n).detach().numpy()

        U = []

        for j in range(len(self.Plant)):
            if U_logist_regression[j] == 0:
                U.append(0.0)
            else:
                U.append(U_logist_regression[j + len(self.Plant)])

        return np.array(U)  # U[i] \in {0%} \cup [60%,100%]


    #
    # def oldController(self, i, delta_t, curve_of_desired_total_current: np.array):
    #     # curve_of_desired_total_current[i] is a reference total CURRENT at time t + i*delta_t
    #
    #     #########
    #     maxOutput_of_electrolyser = 53
    #     #########
    #
    #     hour = 60 * 60
    #     N_hour = int(hour // delta_t)
    #     desired_Rate_at_next_hour = np.average(curve_of_desired_total_current[:N_hour])
    #     # print("desired_Rate_at_next_hour = " + str(desired_Rate_at_next_hour))
    #
    #     Total_output = 0
    #     for i in range(len(self.Plant)):
    #         Total_output += self.Plant[i].getCurrentTarget() * maxOutput_of_electrolyser  # target (set point) current of electrolyser number i
    #
    #     # print("Total_output = " + str(Total_output))
    #
    #     working_elecs = []
    #     not_working_elecs = []
    #     U = [0] * len(self.Plant)
    #     for i in range(len(self.Plant)):
    #         elec = self.Plant[i]
    #         if elec.getCurrentTarget() != 0:
    #             working_elecs.append(elec)
    #         else:
    #             not_working_elecs.append(elec)
    #         U[i] = elec.getCurrentTarget() * 100
    #
    #     # number_of_required_new_electrolysers = 0
    #     # newTotal_output = Total_output
    #
    #     # сортируем на день электролизеры случайно
    #     # если нужно увеличить увеличиваем первый до минимума из его максимума и нехватающей разницы
    #     # если нужно уменьшить, уменьшаем с конца до 60. если все уменьшили до 60 но нужно еще уменьшить , начинаем выключать в обратном порядке
    #
    #     return U  # U[i] \in {0%} \cup [60%,100%]
    #
    #



