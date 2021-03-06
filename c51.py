import random
from collections import deque
import time
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
parser.add_argument("--path", type=str, default='pure_c51', help="save folder path or the test model path")
parser.add_argument("--modelname", type=str, default='testmodel', help="saving model name")
parser.add_argument("--dataset", type=int, default=0, help="choose the model")
parser.add_argument("--vmax", type=int, default=5, help="set the vmax")
parser.add_argument("--vmin", type=int, default=-5, help="set the vmin")
parser.add_argument("--N", type=int, default=51, help="set the numbers of the atoms")
parser.add_argument("--eps", type=float, default=0.25, help="set the epsilon")
parser.add_argument("--gamma", type=float, default=0.5, help="set the gamma")
parser.add_argument("--Lr", type=float, default=0.1, help="set the learning rate")
parser.add_argument("--cap", type=int, default=1, help="the capability of the memory buffer")
parser.add_argument("--step", type=int, default=1, help="the frequency of training")
parser.add_argument("--freq", type=int, default=1, help="the frequency of update the model")
parser.add_argument("--episode", type=int, default=100000, help="set episode rounds")
parser.add_argument("--ucb", type=float, default=0.85, help="set the upper confidence bound")
parser.add_argument("--verbose", action='store_true', default=False, help="print verbose test process")
parser.add_argument("--GPU", action="store_false", default=True, help="use cuda core")
parser.add_argument("--batchsize", type=int, default=1, help="learning batchsize")


# in tabular case state=30, actions=5, agents=3

test_flg = False

class Q_table(nn.Module):
    """
        input should be one hot vector which stands for the states
    """

    def __init__(self, n_states, n_actions, N):
        super(Q_table, self).__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.N = N
        self.Linear = nn.Linear(n_states, N * n_actions, bias=False)
        nn.init.constant(self.Linear.weight, 0.0)

    def forward(self, state):
        par = self.Linear(torch.tensor(state, dtype=torch.float32))
        par = par.reshape(-1, self.n_actions, self.N)
        return f.softmax(par, dim=2)


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
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=alpha)
        self.idx = idx

    def save_checkpoint(self, folder, idx):
        torch.save(self.model.state_dict(), folder + '/agent%d.pkl' % idx)

    def get_opt_action(self, state):
        with torch.no_grad():
            """E = []
            for i in range(self.n_actions):
                E.append(self.target_model(state)[i] * torch.tensor(self.Z, dtype=torch.float32))

            E = torch.vstack(E)
            E = E.sum(dim=1)"""
            Q = self.target_model(state)
            # print(Q.shape)
            Q = Q * torch.tensor(self.Z, dtype=torch.float32)
            Q = torch.squeeze(Q, 0)
            Q = Q.sum(dim=1)

        return Q.argmax()

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
        num_samples = min(batch_size, len(memory))
        replay_samples = random.sample(memory, num_samples)
        # Project Next State Value Distribution (of optimal action) to Current State
        b_s = [sample['s'] for sample in replay_samples]
        b_r = [sample['r'] for sample in replay_samples]
        b_a = [sample['a'] for sample in replay_samples]
        b_s_ = [sample['s_'] for sample in replay_samples]
        b_d_ = [sample['done'] for sample in replay_samples]

        b_s = np.array(b_s)
        b_r = np.array(b_r)
        b_s_ = np.array(b_s_)
        b_a = torch.LongTensor(b_a)

        z_eval = self.model(b_s)  # (batch-size * n_actions * N)
        mb_size = z_eval.size(0)
        # print("b_a shape:{}".format(b_a.shape))
        z_eval = torch.stack([z_eval[i].index_select(dim=0, index=b_a[i, self.idx]) for i in range(mb_size)]).squeeze(1)
        # (batch-size * N)
        z_next = self.target_model(b_s_).detach()  # (m, N_ACTIONS, N_ATOM)
        z_next = z_next.numpy()
        range_value = np.array(self.Z, dtype=float)
        z_range = np.array(self.Z, dtype=float)
        z_range = z_range * self.gamma
        q_next_mean = np.sum(z_next * range_value, axis=2)  # (m, N_ACTIONS)
        opt_act = np.argmax(q_next_mean, axis=1)  # (batch_size)
        opt_act = opt_act.astype(int)
        m_prob = np.zeros([num_samples, self.N])
        global test_flg
        if test_flg:
            print("prev z: {}".format(z_eval))
            print("prev q: {}".format((z_eval*torch.tensor(self.Z)).sum()))
            print("transition: [s:{}, r:{}, a:{}, s_{}, done:{}]".format(b_s[0].argmax(), b_r[0], b_a[0].argmax(), b_s_[0].argmax(), b_d_[0]))
        for i in range(num_samples):
            # Get Optimal Actions for the next states (from distribution z)
            # z = self.model(replay_samples[i]['s_'])  # should be updated model
            # zd = [i.detach() for i in z]  # detach version of model should be updated
            if b_d_[i]:  # Terminal State
                # Distribution collapses to a single point
                Tz = min(self.v_max, max(self.v_min, b_r[i]))
                bj = (Tz - self.v_min) / self.deltaZ
                m_l, m_u = math.floor(bj), math.ceil(bj)
                m_prob[i][int(m_l)] += (m_u - bj)
                m_prob[i][int(m_u)] += (bj - m_l)
            else:
                for j in range(self.N):
                    # print("{} {} {}".format(i, opt_act[i], j))
                    Tz = min(self.v_max,
                             max(self.v_min, b_r[i] + z_range[j]))

                    bj = (Tz - self.v_min) / self.deltaZ
                    m_l, m_u = math.floor(bj), math.ceil(bj)
                    m_prob[i][int(m_l)] += z_next[i, opt_act[i], j] * (m_u - bj)
                    m_prob[i][int(m_u)] += z_next[i, opt_act[i], j] * (bj - m_l)

        m_prob = torch.FloatTensor(m_prob)
        # print("{} {}".format(m_prob.shape, (-torch.log(z_eval + 1e-8)).shape))
        loss = m_prob * (-torch.log(z_eval + 1e-8))
        loss = torch.sum(loss)
        self.optimizer.zero_grad()
        #loss.backward(retain_graph=True)  # ??????????????????
        loss.backward()
        self.optimizer.step()
        if test_flg:
            z_eval = self.model(b_s)  # (batch-size * n_actions * N)
            mb_size = z_eval.size(0)
            # print("b_a shape:{}".format(b_a.shape))
            z_eval = torch.stack(
                [z_eval[i].index_select(dim=0, index=b_a[i, self.idx]) for i in range(mb_size)]).squeeze(1)
            print("learned Z: {}".format(z_eval))
            print("learned q: {}".format((z_eval * torch.tensor(self.Z)).sum()))
        # print('finish')

    def rand_peek(self):
        x = np.zeros([self.n_states])
        state = np.random.randint(0, self.n_states)
        x[state] = 1
        x = torch.FloatTensor(x).reshape(1, -1)
        y = self.model(x).squeeze()
        print("for state {},\n Z is {},\n Q is {}".format(state, y, torch.sum(y * torch.FloatTensor(self.Z))))

    def test_opt_action(self, state, verbose):
        with torch.no_grad():
            Q = self.model(state)
            # print(Q.shape)
            Q = torch.squeeze(Q, 0)
            Q = Q * torch.tensor(self.Z, dtype=torch.float32)
            Q = Q.sum(dim=1)
            action = Q.argmax()
            if verbose:
                print("agent {} q-table is {}, choose action {}".format(self.idx, Q, action))
        return action


start_time = time.time()
training_time = 0

max_r = -100


class Multi_C51:
    """
        multi, independent, C51
    """
    c51agents = []
    memory = deque()

    def __init__(self, n_agents, ucb, n_states, n_actions, N, v_min, v_max, utf, eps, gamma, batch_size=32,
                 alpha=0.001, max_memory=50000, model_name='multi_c51'):
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
        self.max_memory = max_memory
        self.update_target_freq = utf
        self.model_name = model_name

    def get_joint_opt_action(self, state):
        actions = [agent.get_opt_action(state) for agent in self.c51agents]
        return actions

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
        # print("updating")
        for agent in self.c51agents:
            agent.update_target_model()

    def save_checkpoint(self, folder_name, t, verbose):
        Folder = 'logs/' + folder_name
        if not os.path.exists(Folder):  # ???????????????????????????
            os.makedirs(Folder)
        Folder += '/' + str(self.model_name)
        if not os.path.exists(Folder):
            os.makedirs(Folder)
        for idx, agent in zip(range(self.n_agents), self.c51agents):
            agent.save_checkpoint(Folder, idx)
        s = test(self, verbose)
        f = open(Folder + "/result.txt", 'a')
        f.write(s)
        f.close()

    def train_replay(self):
        st_time = time.time()
        for agent in self.c51agents:
            # print("enter agent" + str(agent.idx) + " !!!")
            agent.train_replay(self.memory, self.batch_size)
        global training_time
        training_time += time.time() - st_time

    def test_opt_action(self, state, verbose):
        actions = [agent.test_opt_action(state, verbose) for agent in self.c51agents]
        return actions


def test(multi_c51, verbose):
    env = Env.chooce_the_game(args.dataset)

    R = []
    if verbose:
        print("verbose test process: ")
    for i in range(10):
        ep_r = 0
        s = env.reset()
        if verbose:
            print("episode {}".format(i+1))
        while True:
            a = multi_c51.test_opt_action(s, verbose)  # ??????dqn?????????????????????????????????????????????
            actions_v = []
            for j in range(env.agent_num):
                v = np.zeros(env.action_num)
                v[a[j]] = 1
                actions_v.append(v)
            s_, r, done = env.step(actions_v)  # ??????????????????????????????????????????

            if verbose:
                print("transition(s:{},a:{},r:{},s_:{},dom:{})".format(s.argmax(), a, r, s_.argmax(), done))

            ep_r += r

            if done:
                break
            s = s_  # ?????????????????????????????????????????????
        R.append(ep_r)
    ep_r = 0
    for reward in R:
        ep_r += reward
    ep_r /= 10
    s = "reward:{} \n total mean reward {}\n".format(R, ep_r)

    #print("peek the pi")
    #for i in range(env.agent_num):
    #    print("peeking agent {}".format(i))
    #    multi_c51.c51agents[i].rand_peek()

    print(s)
    s1 = "totol time is %f" % (time.time() - start_time)
    print(s1)
    s += s1 + '\n'
    s2 = "total training time is %f" % training_time
    print(s2)
    s += s2 + '\n'
    return s


def rand_argmax(tens):
    #max_idxs, = torch.where(tens == tens.max())
    # return np.random.choice(max_idxs)
    max_idx = tens.argmax()
    return max_idx


args = parser.parse_args()


def train():
    env = Env.chooce_the_game(args.dataset)
    multi_c51 = Multi_C51(n_agents=env.agent_num, ucb=args.ucb, n_states=env.state_num, n_actions=env.action_num,
                          N=args.N, v_min=args.vmin, v_max=args.vmax, utf=args.freq, eps=args.eps, gamma=args.gamma,
                          max_memory=args.cap, alpha=args.Lr, batch_size=args.batchsize, model_name=args.modelname)

    t = 0
    time_step = args.step
    max_episode = args.episode
    if test_flg:
        max_episode = 1
    for i in range(max_episode):
        s = env.reset()
        ep_r = 0
        while True:
            a = multi_c51.get_joint_action(s)  # ??????dqn?????????????????????????????????????????????
            actions_v = []
            for j in range(env.agent_num):
                v = np.zeros(env.action_num)
                v[a[j]] = 1
                actions_v.append(v)
            s_, r, done = env.step(actions_v)  # ??????????????????????????????????????????
            global max_r
            max_r = max(r, max_r)
            t += 1
            multi_c51.store_transition(s, a, r, s_, done, t)  # dqn???????????????????????????????????????????????????????????????????????????
            # print((s, a, r, s_, done, t))
            ep_r += r

            if t % time_step == 0:
                multi_c51.train_replay()

            if done:
                break
            s = s_  # ?????????????????????????????????????????????

        if i % 1000 == 0:
            print("at episode %d" % i)
            print("max_r = {}".format(max_r))
            multi_c51.save_checkpoint(args.path, t, args.verbose)
            # test(multi_c51, args.verbose)


if __name__ == "__main__":
    print(args)
    if args.train:
        train()
