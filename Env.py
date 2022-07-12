import numpy as np
import pickle


def chooce_the_game(number):
    paths = ["a_mat_model/random_matrix_game_3.pkl", "a_mat_model/random_matrix_game_single.pkl",
             "a_mat_model/random_matrix_game_single_big.pkl", "a_mat_model/random_matrix_game_single_big_2.pkl"]
    return make_mat_game_from_file(paths[number])


def make_mat_game_from_file(filename):
    with open(filename, 'rb') as f:
        matrix_para = pickle.load(f)
        r_mat = matrix_para['reward']
        r_mat = r_mat/5.0  # the upper bound of the absolute value is 1
        trans_mat = matrix_para['trans_mat']
        end_state = np.zeros(np.shape(r_mat)[0])
        state_num = np.shape(r_mat)[0]
        max_episode_length = 40
        init_state = 0
        env = MatrixGame(r_mat, trans_mat, init_state, end_state, max_episode_length, evaluate_mat=True)
        return env


def get_element(array, index):
    ret = array
    # print('array_shape = {}'.format(np.shape(array)))
    # print('index = {}'.format(index))
    for x in index:
        ret = ret[x]
        # print('x = {}, ret_shape = {}'.format(x,np.shape(ret)))
    return ret


class MatrixGame:
    def __init__(self, r_mat, trans_mat, init_state, end_state, max_episode_length, evaluate_mat=False):
        r_shape = np.shape(r_mat)
        self.r_mat = r_mat
        self.trans_mat = trans_mat
        self.state_num = r_shape[0]

        self.agent_num = len(r_shape) - 1
        self.action_num = r_shape[1]
        self.share_observation_space = []
        self.observation_space = []
        self.action_space = []

        self.init_state = init_state
        self.now_state = init_state
        self.step_count = 0
        self.end_state = end_state
        self.max_episode_length = max_episode_length

    def reset(self):
        self.now_state = self.init_state
        self.step_count = 0
        # s = self.get_state()
        return self.now_state

    def get_ac_idx(self, action):
        idx = 0
        for a in action:
            idx = self.action_num * idx + a
            # print('idx = {} a = {}'.format(idx,a))
        return idx

    def get_state(self):
        state = np.zeros(self.state_num)
        state[self.now_state] = 1
        return state

    def step(self, action, evaluate=False):
        # print('step = {} action  = {}'.format(self.step_count))
        sa_index = [self.now_state]
        # sa_index += action
        # action = np.array(action)
        for a in action:
            sa_index.append(np.argmax(a))
        # print('sa_index = {},action = {}'.format(sa_index, action))
        r = get_element(self.r_mat, sa_index)
        next_s_prob = get_element(self.trans_mat, sa_index)
        # print('sa_index = {} next_s_prob = {}'.format(sa_index,next_s_prob))
        next_state = np.random.choice(range(self.state_num), size=1, p=next_s_prob)[0]
        self.now_state = next_state
        self.step_count += 1

        # return self.get_state(), r, self.step_count >= self.max_episode_length
        return self.now_state, r, self.step_count >= self.max_episode_length

    def get_env_info(self):
        env_info = {"n_actions": self.action_num, "n_agents": self.agent_num, "state_shape": self.state_num,
                    "obs_shape": self.state_num, "episode_limit": self.max_episode_length}
        return env_info

    def get_model_info(self, state, action):
        sa_index = [state]
        action = np.array(action)
        # print('action = {}'.format(action))
        for a in action:
            sa_index.append(a)
        r = get_element(self.r_mat, sa_index)
        next_s_prob = get_element(self.trans_mat, sa_index)
        # print('action = {} sa_index = {} self.trans_mat = {} next_s_prob = {}'.format(action,sa_index,self.trans_mat.shape, next_s_prob.shape  ))
        return r,next_s_prob


if __name__ == '__main__':
    env = make_mat_game_from_file("a_mat_model/random_matrix_game_3.pkl")
    print(np.max(env.r_mat))
    print(np.min(env.r_mat))
    print(env.state_num, env.agent_num, env.action_num)
