# -*- coding: utf-8 -*-


class Config:
    iteration = 5
    episode = 300  # 300 games per iteration

    ENV_NAME = "CartPole-v1"
    GAMMA = 0.99  # discount factor for target Q
    INITIAL_EPSILON = 1.0  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    EPSILIN_DECAY = 0.999
    START_TRAINING = 1000  # experience replay buffer size
    BATCH_SIZE = 64  # size of minibatch
    UPDATE_TARGET_NET = 200  # update eval_network params every 200 steps
    LEARNING_RATE = 0.001
    DEMO_RATIO = 0.1
    LAMBDA_1 = 1.0
    LAMBDA_2 = 10e-5
    PRETRAIN_STEPS = 1000
    MODEL_PATH = './model/DQfDDDQN_model'
    replay_buffer_size = 2000

    demo_buffer_size = 500 * 50

class DDQNConfig(Config):
    demo_mode = 'get_demo'


class DQfDConfig(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)
























