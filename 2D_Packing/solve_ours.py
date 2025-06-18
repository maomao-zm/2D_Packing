from ours_brain import DeepQNetwork
from tools.data_process import process_data_xml, process_data_xml_deleteRedundancy
import tools.bottom_left_fill as BLF
import copy
import tools.packing as packing
import tensorflow as tf
from shapely import affinity
from shapely.geometry import Polygon
from patchPanelization import process_patches
'''
综合所有方法，DQN,加入旋转
'''


# np.random.seed(2)  # reproducible
class Solve:
    def __init__(self, name, width, *args, **kwargs):
        self.path = '../Data-xml/' + name + '.xml'
        self.fig_path = 'best_result_picture/' + name + '/S5.png'
        self.width = width
        if 'obj' in args:
            self.polys = process_patches(name)
            polys_DR = process_patches(name)
        else:
            self.polys = process_data_xml(self.path)
            polys_DR = process_data_xml_deleteRedundancy(self.path)

        self.polys_idx = []  ##与state挂钩,,poly index [(idx, angle)]
        ###后序直接获取文件，进行处理
        self.nfp_ass = packing.NFPAssistant(polys_DR, load_history=True, history_path='../record/' + name + '_nfp.csv')
        self.n = len(self.polys)
        self.N_STATES = self.n  # the board contains patches numbers(changed)
        self.angles = [0, 30, 45, 60, 90]  # 可选的旋转角度
        # 动作空间为patch和角度的组合
        self.ACTIONS = [(i, angle) for i in range(self.n) for angle in
                        self.angles]  ## 动作空间为patch和角度的组合，len(ACTIONS) = len(angles)*n

        self.best_polys_idx = []
        self.best_usage = 0.
        if 'compare_usage' in kwargs:
            self.compare_usage = kwargs['compare_usage']

        self.EPSILON = 0.9  # greedy policy
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

        tf.reset_default_graph()  ##因为every times保存了上次运行结束的变量, 所以重置
        self.RL = DeepQNetwork(len(self.ACTIONS), 1, self.ACTIONS,
                               learning_rate=self.ALPHA,
                               reward_decay=self.GAMMA,
                               e_greedy=self.EPSILON,
                               replace_target_iter=200,  ###每200步替换一次target_net的参数
                               memory_size=2000,  ###记忆上限
                               # output_graph=True  ###是否输出tensorboard 文件
                               )

    def get_env_feedback(self, S, A):
        # This is how agent will interact with the environment
        ## if i = n, H ← piece positioning strategy, then ri = C/H, else ri = 0  H == getHeight
        ## A (idx, angle)
        done = False
        if S < self.n - 1:
            self.polys_idx.append(A)
            # tmp_polys = []
            # for (idx, angle) in self.polys_idx:
            #     coords = affinity.rotate(Polygon(self.polys[idx]), angle).exterior.coords[:-1]
            #     rotated_poly = [[coord[0], coord[1]] for coord in coords]
            #     tmp_polys.append(list(rotated_poly))
            # bfl = BLF.BottomLeftFill(self.width, tmp_polys, NFPAssistant=self.nfp_ass)
            # H = bfl.contain_height
            # R = self.C / H
            R = 0
        else:  ##
            self.polys_idx.append(A)
            tmp_polys = []
            for (idx, angle) in self.polys_idx:
                coords = affinity.rotate(Polygon(self.polys[idx]), angle).exterior.coords[:-1]
                rotated_poly = [[coord[0], coord[1]] for coord in coords]
                tmp_polys.append(list(rotated_poly))
            bfl = BLF.BottomLeftFill(self.width, tmp_polys, NFPAssistant=self.nfp_ass)
            H = bfl.contain_height
            R = self.C / H
            done = True
        S_ = S + 1

        return S_, R, done

    def rl(self):  ##run
        step = 0
        for episode in range(self.MAX_EPISODES):
            # initial observation
            observation = 0  ## no patch == state
            self.polys_idx.clear()

            while True:

                # RL choose action based on observation
                action = self.RL.choose_action(observation, self.polys_idx)

                # RL take action and get next observation and reward, done 是否终止
                observation_, reward, done = self.get_env_feedback(observation, action)  ##action = (idx, angle)

                ##DQN 存储记忆
                self.RL.store_transition(observation, action, reward, observation_)

                # 控制学习起始时间和频率(先累计一些记忆再开始学习）
                if (step >= 200) and (step % 5 == 0):
                    self.RL.learn()

                # swap observation, state_ = state
                observation = observation_

                # break while loop when end of this   episode
                if done:
                    break
                step += 1

            if observation == self.n:  ##terminal
                print("Episode", episode, ": the sequence is: ", self.polys_idx)
                tmp_polys = []
                for (idx, angle) in self.polys_idx:
                    coords = affinity.rotate(Polygon(self.polys[idx]), angle).exterior.coords[:-1]
                    rotated_poly = [[coord[0], coord[1]] for coord in coords]
                    tmp_polys.append(list(rotated_poly))
                bfl = BLF.BottomLeftFill(self.width, tmp_polys, NFPAssistant=self.nfp_ass)
                bfl.textPrint()
                usage = bfl.patches_area / bfl.board_area
                print("The Usage Percentage:", usage)
                if usage > self.best_usage:
                    self.best_usage = copy.deepcopy(usage)
                    self.best_polys_idx = copy.deepcopy(self.polys_idx)
                if usage > self.compare_usage:
                    self.compare_usage = copy.deepcopy(usage)
                    bfl.only_savefig(fig_path=self.fig_path)

        # self.plot_cost()  ##cost曲线

    def plot_cost(self):
        self.RL.plot_cost()
