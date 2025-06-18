import tools.bottom_left_fill as BLF
import pandas as pd
import numpy as np
import copy
from tools.data_process import process_data_xml, process_data_xml_deleteRedundancy
import tools.packing as packing
from patchPanelization import process_patches
'''
0，reward 原采用 高度的倒数且完整序列packing之后reward
改进点：1，放入每一个patch之后reward
3，使用DQN代替传统方法
4, 综合以上方法
'''
# np.random.seed(2)  # reproducible

class Solve:
    def __init__(self, name, width, *args, **kwargs):
        self.path = '../Data-xml/' + name + '.xml'
        self.fig_path = 'best_result_picture/' + name + '/S0.png'
        self.width = width
        if 'obj' in args:
            self.polys = process_patches(name)
            polys_DR = process_patches(name)
        else:
            self.polys = process_data_xml(self.path)
            polys_DR = process_data_xml_deleteRedundancy(self.path)
        # self.polys = process_data_xml(self.path)
        self.polys_idx = [] ##与state挂钩,,poly index
        # polys_DR = process_data_xml_deleteRedundancy(self.path)
        ###后序直接获取文件，进行处理
        self.nfp_ass = packing.NFPAssistant(polys_DR, load_history=True, history_path='../record/' + name +'_nfp.csv')

        self.n = len(self.polys)
        self.N_STATES = self.n  # the board contains patches numbers(changed)
        self.ACTIONS = [i for i in range(self.n)]  # the patches index

        self.best_polys_idx = []
        self.best_usage = 0.
        if 'compare_usage' in kwargs:
            self.compare_usage = kwargs['compare_usage']

        self.EPSILON = 0.9  # greedy police
        self.ALPHA = 0.5  # learning rate
        self.GAMMA = 1.  # discount factor
        self.MAX_EPISODES = 300  # maximum episodes
        self.C = 100  ## the constant

        if 'EPSILON' in kwargs:
            self.EPSILON = kwargs['EPSILON']
        if 'ALPHA' in kwargs:
            self.ALPHA = kwargs['ALPHA']
        if 'GAMMA' in kwargs:
            self.GAMMA = kwargs['GAMMA']
        if 'MAX_EPISODES' in kwargs:
            self.MAX_EPISODES = kwargs['MAX_EPISODES']
        if 'C' in kwargs:
            self.C = kwargs['C']

    def build_q_table(self):  ## the idx is int
        table = pd.DataFrame(
            np.zeros((self.N_STATES, len(self.ACTIONS)))
        )
        # print(table)    # show table
        return table


    def choose_action(self, state, q_table):  ##e
        # This is how to choose an action
        state_actions = q_table.iloc[state, :]  ##定位到对应的action那一行
        tmp = list(set(self.ACTIONS) - set(self.polys_idx))  ##求差集得到的是在a里但不在b里的数据集合，a独有 ##set{}

        if (np.random.uniform() > self.EPSILON) or (
        (state_actions == 0).all()):  # act non-greedy or state-action have no value
            action_name = np.random.choice(tmp)
        else:  # act greedy
            tmp_list = state_actions.tolist()
            tmp_max = float('-inf')  ## min value
            action_name = -1
            for i in range(len(tmp_list)):
                if i in set(self.polys_idx):
                    continue
                if tmp_list[i] > tmp_max:
                    tmp_max = tmp_list[i]
                    action_name = i
                # action_name = state_actions.idxmax()   ## just idx, int # replace argmax to idxmax as argmax means a different function in newer version of pandas

        return action_name  ## int

    def get_env_feedback(self, S, A):
        # This is how agent will interact with the environment
        ## if i = n, H ← piece positioning strategy, then ri = C/H, else ri = 0  H == getHeight

        R = 0
        if S < self.n - 1:
            self.polys_idx.append(A)
        else:
            self.polys_idx.append(A)
            tmp_polys = [self.polys[i] for i in self.polys_idx]
            blf = BLF.BottomLeftFill(self.width, tmp_polys, NFPAssistant=self.nfp_ass)
            H = blf.contain_height
            R = self.C / H
            # blf.showAll()
        S_ = S + 1

        return S_, R

    def update_env(self, S, episode):
        # This is how environment be updated
        if S == 0:  ## no patch
            self.polys_idx.clear()
        if S == self.n:  ##terminal
            print("Episode", episode, ": the sequence is: ", self.polys_idx)
            tmp_polys = [self.polys[i] for i in self.polys_idx]
            blf = BLF.BottomLeftFill(self.width, tmp_polys, NFPAssistant=self.nfp_ass)
            blf.textPrint()
            usage = blf.patches_area / blf.board_area
            print("The Usage Percentage:", usage)
            if usage > self.best_usage:
                self.best_usage = copy.deepcopy(usage)
                self.best_polys_idx = copy.deepcopy(self.polys_idx)

            if usage > self.compare_usage:
                self.compare_usage = copy.deepcopy(usage)
                blf.only_savefig(fig_path=self.fig_path)

    def rl(self):
        # main part of RL loop
        q_table = self.build_q_table()
        for episode in range(self.MAX_EPISODES):
            S = 0  ##初始无patch
            is_terminated = False
            self.update_env(S, episode)
            while not is_terminated:
                A = self.choose_action(S, q_table)
                S_, R = self.get_env_feedback(S, A)  # take action & get next state and reward
                q_predict = q_table.iloc[S, A]
                if S_ != self.n:
                    q_target = R + self.GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
                else:
                    q_target = R  # next state is terminal
                    is_terminated = True  # terminate this episode

                q_table.iloc[S, A] += self.ALPHA * (q_target - q_predict)  # update
                S = S_  # move to next state
                self.update_env(S, episode)

        return q_table
