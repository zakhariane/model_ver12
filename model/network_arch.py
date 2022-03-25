
# from keras.layers import Input, Flatten, Dense
# from keras.models import Model
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_function(x, k):
    return k*x

# def mynorm(x):
#
#     return x / np.linalg.norm(x.detach().numpy())

class DNetwork(nn.Module):
    def __init__(self):
        super(DNetwork, self).__init__()

        state_len = 99
        num_elecs = 5

        hiden_layer_len_1 = 50
        hiden_layer_len_2 = 50

    
        self.hidden_1 = nn.Linear(state_len, hiden_layer_len_1, bias=True)
        self.hidden_2 = nn.Linear(hiden_layer_len_1, hiden_layer_len_2, bias=True)
  
        self.output = nn.Linear(hiden_layer_len_2, 2 * num_elecs, bias=True)

        
        self.activation_on_hidden_1 = nn.ReLU()
        self.activation_on_hidden_2 = linear_function
        
        self.activation_on_output = linear_function

        # self.num_filter1 = 8
        # self.num_filter2 = 16
        # self.num_padding = 2
        # # input is 28x28
        # # padding=2 for same padding
        # self.conv1 = nn.Conv2d(1, self.num_filter1, 5, padding=self.num_padding)
        # # feature map size is 14*14 by pooling
        # # padding=2 for same padding
        # self.conv2 = nn.Conv2d(self.num_filter1, self.num_filter2, 5, padding=self.num_padding)
        # # feature map size is 7*7 by pooling
        # self.fc = nn.Linear(self.num_filter2 * 7 * 7, 10)

        self.OUT = []

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_filter2 * 7 * 7)  # reshape Variable
        # x = self.fc(x)

        x = self.hidden_1(x)
        x = self.activation_on_hidden_1(x)
        x = self.hidden_2(x)
        x = self.activation_on_hidden_2(x, 1)
        x = self.output(x)
        x = self.activation_on_output(x, 1)

        self.OUT = []
        for x_j in x.detach().numpy():
            self.OUT.append(x_j)

        n2 = len(x)

        assert n2 == 10

        regression_half = torch.narrow(x, dim=0, start= n2//2, length=n2//2)
        norm_of_regression_half1 = np.linalg.norm(regression_half.detach().numpy())

        for j in range(n2):

            # assert abs(x[j]) <= 1
            #
            # if x[j] <= -0.999:
            #     x[j] = 0
            # elif x[j] >= 0.999:
            #     x[j] = 100
            # else:
            #     x[j] = 20.02002 * (x[j]+1) + 60

            if j < n2 // 2:
                if x[j] <= 0:
                    x[j] = 0
                else:
                    x[j] = 1

            else:
                x[j] = 20 * (x[j] / norm_of_regression_half1 + 1) + 60


        return x

    # def build(self):
    #     input_layer = Input(shape=(28, 28))
    #     x = Flatten()(input_layer)
    #     x = Dense(units=200, activation='relu')(x)
    #     # x = Dense(units=150, activation = 'relu')(x)
    #     output_layer = Dense(units=10, activation='softmax')(x)
    #
    #     model = Model(input_layer, output_layer)
    #
    #     return model

    # def train(self, input, output):
    #     self.model.fit(input, output,
    #                    shuffle=False,
    #                    epochs=1,
    #                    batch_size=len(input))


class Agent:
    def __init__(self):
        self.model = DNetwork()

        orig_model = copy.deepcopy(self.model)

        self.model_shapes = []
        for param in orig_model.parameters():
            p = param.data.cpu().numpy()
            self.model_shapes.append(p.shape)

    def updateParams(self, flat_param: np.array):
        idx = 0
        i = 0

        for param in self.model.parameters():
            delta = np.product(self.model_shapes[i])
            block = flat_param[idx:idx + delta]
            block = np.reshape(block, self.model_shapes[i])
            i += 1
            idx += delta
            block_data = torch.from_numpy(block).float()

            param.data = block_data


    def getAction(self, state, i, n):

        # W = np.random.rand(n, state.shape[0])
        #
        # action = np.dot(W, state)
        #
        # for j in range(n):
        #     action[j] -= 390
        #
        #     if action[j] < 60:
        #         action[j] = 0
        #     elif action[j] > 100:
        #         action[j] = 100


        #action = self.model.forward(state)

        action = self.model(torch.from_numpy(state).float())

        #assert abs(max(state) - 1) < 0.1

        if abs(max(state) - 1) > 0.1:
            print('========================== max state > 0.1 ==============================')
            print(max(state))
            print(np.argmax(state))


        # if i % int(15* 60 / 1) == 0: # delta_t = 1
        #     print('State incoming in NET and action ===============  ' + str(i))
        #     print(state.shape)
        #     print(state)
        #     print()
        #     print('max  =  ' + str(max(state)))
        #     print('sum  =  ' + str(sum(state)))
        #     print()
        #     if True: # max(self.model.OUT) >= 0.999 or min(self.model.OUT) <= -0.999:
        #         print(action)
        #         print(self.model.OUT)
        #     print('State incoming in NET and action ===============  ' + str(i))

        return action

    #=========

    def setWeights_fromFile(self, filepath):
        self.model.load_weights(filepath)

    def saveWeights_toFile(self, filepath):
        self.model.save_weights(filepath)

