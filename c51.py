import random
from collections import deque

import torch
import Env
from torch import nn
from torch.nn import functional as f
import argparse
import numpy as np
import os
import math

parser = argparse.ArgumentParser(description="test c51 for the independence multi-agent environment")
parser.add_argument("--train", action="store_true", default=True, help="train the model")
parser.add_argument("--test", action="store_true", default=False, help="test the model")
parser.add_argument("--path", type=str, default='test', help="save folder path or the test model path")
parser.add_argument("--dataset", type=int, default=0, help="choose the model")
parser.add_argument("--vmax", type=int, default=50, help="set the vmax")
parser.add_argument("--vmin", type=int, default=-50, help="set the vmin")
parser.add_argument("--N", type=int, default=51, help="set the numbers of the atoms")
parser.add_argument("--eps", type=float, default=0.1, help="set the epsilon")
parser.add_argument("--gamma", type=float, default=0.75, help="set the gamma")
parser.add_argument("--alpha", type=float, default=0.005, help="set the learning rate")
parser.add_argument("--capacity", type=int, default=10000, help="the capability of the memory buffer")
parser.add_argument("--step", type=int, default=300, help="the frequency of training")
parser.add_argument("--freq", type=int, default=1000, help="the frequency of update the model")
parser.add_argument("--episode", type=int, default=20000, help="set episode rounds")
parser.add_argument("--ucb", type=float, default=0.85, help="set the upper confidence bound")


class Q_table(nn.Module):
    def __init__(self, n_states, n_actions, N):
        super(Q_table, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.N = N
        self.par_list = nn.ParameterList()
        for i in range(n_states * n_actions):
            self.par_list.append(nn.Parameter(torch.zeros(size=[N])))
        # print(self.par_list)

    def forward(self, state):
        l = []
        for i in range(self.n_actions):
            # print(state)
            l.append(f.softmax(self.par_list[state * self.n_actions + i]))
        return l


class C51agent:
    def __init__(self, n_states, n_actions, N, v_min, v_max, eps, gamma, alpha, idx):
        self.n_states = n_states
        self.n_actions = n_actions
        self.N = N
        self.v_min = v_min
        self.v_max = v_max
        self.model = Q_table(n_states, n_actions, N)
        self.target_model = Q_table(n_states, n_actions, N)
        self.eps = eps
        self.deltaZ = (v_max - v_min) / float(N - 1)
        self.Z = [v_min + i * self.deltaZ for i in range(N)]
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)
        self.idx = idx

    def save_checkpoint(self, folder, idx):
        torch.save(self.model.state_dict(), folder + '/model%d.pkl' % idx)

    def get_opt_action(self, state):
        with torch.no_grad():
            E = []
            for i in range(self.n_actions):
                E.append(self.target_model(state)[i] * torch.tensor(self.Z, dtype=torch.float32))

            E = torch.vstack(E)
            E = E.sum(dim=1)
        return rand_argmax(E)

    def get_action(self, state):
        rand = torch.rand(1)
        if rand <= self.eps:
            return random.randrange(0, self.n_actions)
        else:
            return self.get_opt_action(state)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_replay(self, memory, batch_size):
        # print("enter here")
        num_samples = min(batch_size * 40, len(memory))
        replay_samples = random.sample(memory, num_samples)
        # Project Next State Value Distribution (of optimal action) to Current State
        for i in range(num_samples):
            loss = torch.zeros(1, dtype=torch.float32)
            m_prob = torch.zeros([self.N], dtype=torch.float32, requires_grad=False)
            # Get Optimal Actions for the next states (from distribution z)
            z = self.model(replay_samples[i]['s_'])
            z_1 = self.model(replay_samples[i]['s_'])
            z_ = [i.detach() for i in z_1]
            opt_action = self.get_opt_action(replay_samples[i]['s_'])
            if replay_samples[i]['done']:  # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, replay_samples[i]['r']))
                bj = (Tz - self.v_min) / self.deltaZ
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[int(m_l)] += (m_u - bj)
                m_prob[int(m_u)] += (bj - m_l)
            else:
                for j in range(self.N):
                    Tz = min(self.v_max, max(self.v_min, replay_samples[i]['r'] + self.gamma * z[opt_action][j]).detach())
                    bj = (Tz - self.v_min) / self.deltaZ
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[int(m_l)] += z_[opt_action][j] * (m_u - bj)
                    m_prob[int(m_u)] += z_[opt_action][j] * (bj - m_l)
            rz = self.model(replay_samples[i]['s'])[replay_samples[i]['a'][self.idx]]
            """a = replay_samples[i]['a'][self.idx]
            acts = torch.full([1, self.N], a, dtype=torch.int64)
            rz = rz.gather(0, acts).reshape(51)"""

            # print(replay_samples[i])
            # print("rz:")
            # print(rz)
            loss += f.kl_div(torch.log(rz), m_prob)
            # print("loss:")
            # print(loss)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)  # 误差反向传播
            self.optimizer.step()
            # print('finish')


class Multi_C51:
    """
        multi, independent, C51
    """
    c51agents = []
    memory = deque()

    def __init__(self, n_agents, ucb, n_states, n_actions, N, v_min, v_max, utf, eps, gamma, batch_size=32,
                 alpha=0.001):
        self.n_agents = n_agents
        self.n_actions = n_agents
        self.n_states = n_agents
        self.N = N
        self.v_min = v_min
        self.v_max = v_max
        self.batch_size = batch_size
        for i in range(n_agents):
            self.c51agents.append(C51agent(n_states, n_actions, N, v_min, v_max, eps, gamma, alpha, i))
        self.ucb = ucb
        self.max_memory = 50000
        self.update_target_freq = utf

    def get_joint_action(self, state):
        actions = [agent.get_action(state) for agent in self.c51agents]
        return actions

    def store_transition(self, s, a, r, s_, done, t):
        self.memory.append({'s': s, 'a': a, 'r': r, 's_': s_, 'done': done})

        if len(self.memory) > self.update_target_freq:
            self.memory.popleft()

        if t % self.update_target_freq == 0:
            self.update_target_models()

    def update_target_models(self):
        print("updating")
        for agent in self.c51agents:
            print(agent.model(0))
            agent.update_target_model()

    def save_checkpoint(self, folder_name, t):
        Folder = 'logs/' + folder_name
        if not os.path.exists(Folder):  # 是否存在这个文件夹
            os.makedirs(Folder)
        Folder += '/' + str(t)
        if not os.path.exists(Folder):
            os.makedirs(Folder)
        for idx, agent in zip(range(self.n_agents), self.c51agents):
            agent.save_checkpoint(Folder, idx)

    def train_replay(self):
        for agent in self.c51agents:
            # print("enter agent" + str(agent.idx) + " !!!")
            agent.train_replay(self.memory, self.batch_size)


def rand_argmax(tens):
    max_idxs, = torch.where(tens == tens.max())
    return np.random.choice(max_idxs)


args = parser.parse_args()


def train():
    env = Env.chooce_the_game(args.dataset)
    multi_c51 = Multi_C51(env.agent_num, args.ucb, env.state_num, env.action_num,
                          args.N, args.vmin, args.vmax, args.freq, args.eps, args.gamma)

    t = 0
    time_step = args.step
    for i in range(args.episode):
        s = env.reset()
        ep_r = 0
        while True:
            a = multi_c51.get_joint_action(s)  # 根据dqn来接受现在的状态，得到一个行为
            s_, r, done = env.step(a)  # 根据环境的行为，给出一个反馈

            t += 1
            multi_c51.store_transition(s, a, r, s_, done, t)  # dqn存储现在的状态，行为，反馈，和环境导引的下一个状态
            # print((s, a, r, s_, done, t))
            ep_r += r

            if t % time_step == 0:
                multi_c51.train_replay()

            if done:
                break
            s = s_  # 现在的状态赋值到下一个状态上去

        if i % 100 == 0:
            multi_c51.save_checkpoint(args.path, t)
            print("at episode %d, with total reward %f" % (i, ep_r))


if __name__ == "__main__":
    print(args)
    if args.train:
        train()
