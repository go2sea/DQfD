# -*- coding: utf-8 -*
import gym
from gym import wrappers
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
from collections import deque
import pickle
import copy

# Hyper Parameters for DQN
GAMMA = 0.99  # discount factor for target Q
INITIAL_EPSILON = 1.0  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
EPSILIN_DECAY = 0.999
START_TRAINING = 1000  # experience replay buffer size
BATCH_SIZE = 64  # size of minibatch
UPDATE_TARGET_NETwORK = 1000  # update eval_network params every 200 steps
LEARNING_RATE = 0.001
DEMO_RATIO = 0.1
LAMBDA_1 = 1.0
LAMBDA_2 = 10e-5
PRETRAIN_STEPS = 1000
MODEL_PATH = 'C:\\Users\\go2sea\\Desktop\\savers\\saver_Double_DQN\\model.ckpt'


class DQfD_DDQN():
    # DQN Agent
    def __init__(self, env, demo_mode):
        self.sess = tf.InteractiveSession()
        # init experience replay
        self.replay_buffer = deque(maxlen=2000)  # store the item generated from select_network
        self.demo_buffer = deque()  # store the demo data
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.demo_mode = demo_mode

        self.create_network()
        self.create_training_method()

        self.sess.run(tf.global_variables_initializer())

    # use the expert-demo-data to pretrain
    def pre_train(self):
        print('pre-training ...')
        for i in range(PRETRAIN_STEPS):
            if i % 200 == 0:
                print(i, 'th step of pre-trianing ...')
            self.train_Q_network(pre_train=True)
        self.time_step = 0
        print('pre-train finish ...')

    def create_network(self):
        def build_layers(state, c_names, n_l1, n_l2, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.state_dim, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(state, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w21', [n_l1, n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [n_l2, self.action_dim], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.action_dim], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3

            return out

        # --------------------- build select network ----------------------
        self.select_input = tf.placeholder("float", [None, self.state_dim])
        with tf.variable_scope('select_net'):
            c_names = ['select_net_params', tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.REGULARIZATION_LOSSES]  # 注意：只有select网络的参数添加到正则化collection中
            w_initializer = tf.random_uniform_initializer(-0.1, 0.1)
            b_initializer = tf.constant_initializer(0.1)
            self.Q_select = build_layers(self.select_input, c_names, 24, 24, w_initializer, b_initializer)
        # --------------------- build eval network ----------------------
        self.eval_input = tf.placeholder("float", [None, self.state_dim])
        with tf.variable_scope('eval_net'):
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_uniform_initializer(-0.1, 0.1)
            b_initializer = tf.constant_initializer(0.1)
            self.Q_eval = build_layers(self.eval_input, c_names, 24, 24, w_initializer, b_initializer)

    def save_model(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, MODEL_PATH)
        print("Model saved in file: ", MODEL_PATH)

    def restore_model(self):
        saver = tf.train.Saver()
        saver.restore(self.sess, MODEL_PATH)
        print("Model restored.")

    def loss_l(self, state, ae, a):
        return 0.0 if ae == a else 0.8

    def loss_jeq(self, Q_select):
        jeq = 0.0
        for i in range(BATCH_SIZE):
            ae = self.action_batch[i]
            max_value = float("-inf")
            for a in range(self.action_dim):
                max_value = tf.maximum(Q_select[i][a] + self.loss_l(self.select_input[i], ae, a), max_value)
            jeq += self.isdemo[i] * (max_value - Q_select[i][ae])
        return jeq

    def create_training_method(self):
        self.action_batch = tf.placeholder("int32", [None])
        self.y_input = tf.placeholder("float", [None, self.action_dim])
        self.isdemo = tf.placeholder("float", [None])
        Q_select = self.Q_select
        loss_dq = tf.reduce_mean(tf.squared_difference(Q_select, self.y_input))
        loss_jeq = self.loss_jeq(Q_select)
        loss_l2 = tf.reduce_sum([tf.reduce_mean(reg_loss) for reg_loss in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)])

        if self.demo_mode == 'get_demo':
            self.loss = loss_dq + LAMBDA_2 * loss_l2
        else:
            self.loss = loss_dq + LAMBDA_1 * loss_jeq + LAMBDA_2 * loss_l2

        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)  # optimizer只更新selese_network中的参数

    def perceive(self, state, action, reward, next_state, done, demo):
        # epsilon是不断变小的，也就是随机性不断变小:开始需要更多的探索，所以动作偏随机，之后需要动作能够有效，因此减少随机。
        self.epsilon = max(FINAL_EPSILON, self.epsilon * EPSILIN_DECAY)
        # 经验池添加
        self.replay_buffer.append((state, action, reward, next_state, done, demo))

    def train_Q_network(self, pre_train=False):
        if not pre_train and len(self.replay_buffer) < START_TRAINING:
            return
        self.time_step += 1
        # 经验池随机采样minibatch
        if pre_train:
            minibatch = random.sample(self.demo_buffer, BATCH_SIZE)
        elif self.demo_mode == 'get_demo':
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        elif self.demo_mode == 'use_demo':
            demo_num = int(BATCH_SIZE * DEMO_RATIO)
            minibatch = random.sample(self.replay_buffer, BATCH_SIZE - demo_num)
            demo_batch = random.sample(self.demo_buffer, demo_num)
            minibatch.extend(demo_batch)
        else:
            assert(False)

        np.random.shuffle(minibatch)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done = [data[4] for data in minibatch]
        isdemo = [data[5] for data in minibatch]

        Q_select = self.Q_select.eval(feed_dict={self.select_input: next_state_batch})
        Q_eval = self.Q_eval.eval(feed_dict={self.eval_input: next_state_batch})

        # convert true to 1, false to 0
        done = np.array(done) + 0

        y_batch = np.zeros((BATCH_SIZE, self.action_dim))
        for i in range(0, BATCH_SIZE):
            temp = self.Q_select.eval(feed_dict={self.select_input: state_batch[i].reshape((-1, 4))})[0]
            action = np.argmax(Q_select[i])
            temp[action_batch[i]] = reward_batch[i] + (1 - done[i]) * GAMMA * Q_eval[i][action]
            y_batch[i] = temp

        # 新产生的样本输入
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.select_input: state_batch,
            self.action_batch: action_batch,
            self.isdemo: isdemo
        })

    def egreedy_action(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(self.Q_select.eval(feed_dict={self.select_input: [state]})[0])

    def update_target_network(self):
        select_params = tf.get_collection('select_net_params')
        eval_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(e, s) for e, s in zip(eval_params, select_params)])


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = "CartPole-v1"
EPISODE = 5000  # Episode limitation


def map(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def run_DQfD(index, episode_limit, env):
    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfD_DDQN(env, 'use_demo')
    with open('/Users/mahailong/DQfD/demo.p', 'rb') as f:
        agent.demo_buffer = pickle.load(f)
    # use the demo data to pre-train network
    agent.pre_train()
    scores = []
    for e in range(EPISODE):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive(state, action, reward, next_state, done, 0.0)
            agent.train_Q_network(pre_train=False)
            state = next_state

        if done:
            scores.append(score)
            agent.update_target_network()
            print("episode:", e, "  score:", score, "  memory length:", len(agent.replay_buffer), "  epsilon:",
                  agent.epsilon)
            # stop traning when the mean of scores of the last 10 episode is bigger than 495
            # if np.mean(scores[-min(10, len(scores)):]) > 495:
            #     break

            if e == episode_limit:
                break
    return scores


def run_DDQN(index, episode_limit, env):
    with tf.variable_scope('DDQN_' + str(index)):
        agent = DQfD_DDQN(env, 'get_demo')
    scores = []
    for e in range(EPISODE):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        demo = []
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive(state, action, reward, next_state, done, 0.0)  # 0. means it is not a demo data
            demo.append((state, action, reward, next_state, done, 1.0))  # record the data that could be expert-data
            agent.train_Q_network(pre_train=False)
            state = next_state

        if done:
            scores.append(score)
            agent.update_target_network()
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            # stop traning when the mean of scores of the last 10 episode is bigger than 490
            # if np.mean(scores[-min(10, len(scores)):]) > 490:
            #     break
            if e == episode_limit:
                break
    return scores


# get expert demo data
def get_demo_data(env):
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # agent.restore_model()
    with tf.variable_scope('get_demo_data'):
        agent = DQfD_DDQN(env, 'get_demo')
    scores = []
    for e in range(EPISODE):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        demo = []
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            # Define reward for agent
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive(state, action, reward, next_state, done, 0.0)  # 0. means it is not a demo data
            demo.append((state, action, reward, next_state, done, 1.0))  # record the data that could be expert-data
            # print("len(demo):", len(demo))
            agent.train_Q_network(pre_train=False)
            state = next_state

        if done:
            scores.append(score)
            if score == 500:  # expert demo data
                agent.demo_buffer.extend(demo)
            agent.update_target_network()
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            # stop traning when the mean of scores of the last 10 episode is bigger than 490
            # if np.mean(scores[-min(10, len(scores)):]) > 490:
            #     break
            if len(agent.demo_buffer) >= 500 * 50:
                break

    # write the demo data to a file
    with open('/Users/mahailong/DQfD/test_demo.p', 'wb') as f:
        pickle.dump(agent.demo_buffer, f, protocol=2)


def main():
    env = gym.make(ENV_NAME)
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)

    iteration = 5
    episode_limit = 300

    # ------------------------ get demo scores by DDQN -----------------------------
    # get_demo_data(env)
    # --------------------------  get DDQN scores ----------------------------------
    ddqn_sum_scores = np.zeros(episode_limit)
    for i in range(iteration):
        scores = run_DDQN(i, episode_limit, env)
        for e in range(episode_limit):
            ddqn_sum_scores[e] += scores[e]
    ddqn_mean_scores = ddqn_sum_scores / iteration
    # write the scores to a file
    with open('/Users/mahailong/DQfD/ddqn_mean_scores.p', 'wb') as f:
        pickle.dump(ddqn_mean_scores, f, protocol=2)
    # ----------------------------- get DQfD scores --------------------------------
    dqfd_sum_scores = np.zeros(episode_limit)
    for i in range(iteration):
        scores = run_DQfD(i, episode_limit, env)
        for e in range(episode_limit):
            dqfd_sum_scores[e] += scores[e]
    dqfd_mean_scores = dqfd_sum_scores / iteration
    # write the scores to a file
    with open('/Users/mahailong/DQfD/dqfd_mean_scores.p', 'wb') as f:
        pickle.dump(dqfd_mean_scores, f, protocol=2)

    map(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores, xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')

    # env.close()
    # gym.upload('/tmp/carpole_DDQN-1', api_key='sk_VcAt0Hh4RBiG2yRePmeaLA')


if __name__ == '__main__':
    main()