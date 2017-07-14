# -*- coding: utf-8 -*
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from gym import wrappers
import gym
import numpy as np
import pickle
from Config import Config, DDQNConfig, DQfDConfig
from DQfDDDQN import DQfDDDQN

def map_scores(dqfd_scores=None, ddqn_scores=None, xlabel=None, ylabel=None):
    if dqfd_scores is not None:
        plt.plot(dqfd_scores, 'r')
    if ddqn_scores is not None:
        plt.plot(ddqn_scores, 'b')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.show()


def run_DDQN(index, env):
    with tf.variable_scope('DDQN_' + str(index)):
        agent = DQfDDDQN(env, DDQNConfig())
    scores = []
    for e in range(Config.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive(state, action, reward, next_state, done, 0.0)  # 0. means it is not a demo data
            agent.train_Q_network(pre_train=False, update=False)
            state = next_state
        if done:
            scores.append(score)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            # if np.mean(scores[-min(10, len(scores)):]) > 490:
            #     break
    return scores


def run_DQfD(index, env):
    with tf.variable_scope('DQfD_' + str(index)):
        agent = DQfDDDQN(env, DQfDConfig())
    with open(Config.DEMO_DATA_PATH, 'rb') as f:
        agent.demo_buffer = pickle.load(f)
    agent.pre_train()  # use the demo data to pre-train network
    scores = []
    for e in range(Config.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive(state, action, reward, next_state, done, 0.0)
            agent.train_Q_network(pre_train=False, update=False)
            state = next_state
        if done:
            scores.append(score)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  memory length:", len(agent.replay_buffer), "  epsilon:",
                  agent.epsilon)
            # if np.mean(scores[-min(10, len(scores)):]) > 495:
            #     break
    return scores


# get expert demo data
def get_demo_data(env):
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # agent.restore_model()
    with tf.variable_scope('get_demo_data'):
        agent = DQfDDDQN(env, 'get_demo')
    for e in range(Config.episode):
        done = False
        score = 0  # sum of reward in one episode
        state = env.reset()
        demo = []
        while done is False:
            action = agent.egreedy_action(state)  # e-greedy action for train
            next_state, reward, done, _ = env.step(action)
            score += reward
            reward = reward if not done or score == 499 else -100
            agent.perceive(state, action, reward, next_state, done, 0.0)  # 0. means it is not a demo data
            demo.append((state, action, reward, next_state, done, 1.0))  # record the data that could be expert-data
            agent.train_Q_network(pre_train=False, update=False)
            state = next_state
        if done:
            if score == 500:  # expert demo data
                agent.demo_buffer.extend(demo)
            agent.sess.run(agent.update_target_net)
            print("episode:", e, "  score:", score, "  demo_buffer:", len(agent.demo_buffer),
                  "  memory length:", len(agent.replay_buffer), "  epsilon:", agent.epsilon)
            if len(agent.demo_buffer) >= Config.demo_buffer_size:
                agent.demo_buffer = agent.demo_buffer[:Config.demo_buffer_size]
                break
    # write the demo data to a file
    with open(Config.DEMO_DATA_PATH, 'wb') as f:
        pickle.dump(agent.demo_buffer, f, protocol=2)


if __name__ == '__main__':
    env = gym.make(Config.ENV_NAME)
    # env = wrappers.Monitor(env, '/tmp/CartPole-v0', force=True)
    # ------------------------ get demo scores by DDQN -----------------------------
    # get_demo_data(env)
    # --------------------------  get DDQN scores ----------------------------------
    ddqn_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DDQN(i, env)
        ddqn_sum_scores = [a + b for a, b in zip(scores, ddqn_sum_scores)]
    ddqn_mean_scores = ddqn_sum_scores / Config.iteration
    with open('/Users/mahailong/DQfD/ddqn_mean_scores.p', 'wb') as f:
        pickle.dump(ddqn_mean_scores, f, protocol=2)
    # ----------------------------- get DQfD scores --------------------------------
    dqfd_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DQfD(i, env)
        dqfd_sum_scores = [a + b for a, b in zip(scores, dqfd_sum_scores)]
    dqfd_mean_scores = dqfd_sum_scores / Config.iteration
    with open('/Users/mahailong/DQfD/dqfd_mean_scores.p', 'wb') as f:
        pickle.dump(dqfd_mean_scores, f, protocol=2)

    # map_scores(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores,
        # xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
    # env.close()
    # gym.upload('/tmp/carpole_DDQN-1', api_key='sk_VcAt0Hh4RBiG2yRePmeaLA')


