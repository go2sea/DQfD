# -*- coding: utf-8 -*-
import os


class Config:
    ENV_NAME = "CartPole-v1"
    GAMMA = 0.99  # discount factor for target Q
    INITIAL_EPSILON = 1.0  # starting value of epsilon
    FINAL_EPSILON = 0.01  # final value of epsilon
    EPSILIN_DECAY = 0.999
    START_TRAINING = 1000  # experience replay buffer size
    BATCH_SIZE = 64  # size of minibatch
    UPDATE_TARGET_NET = 10000  # update eval_network params every 200 steps
    LEARNING_RATE = 0.001
    DEMO_RATIO = 0.1
    LAMBDA = [1.0, 0.0, 1.0, 10e-5]  # for [loss_dq, loss_n_dq, loss_jeq, loss_l2]
    PRETRAIN_STEPS = 5000  # 750000
    MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/DQfD_model')
    DEMO_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'demo.p')

    demo_buffer_size = 500 * 50
    replay_buffer_size = demo_buffer_size * 2
    iteration = 5
    episode = 300  # 300 games per iteration
    trajectory_n = 10  # for n-step TD-loss (both demo data and generated data)


class DDQNConfig(Config):
    demo_mode = 'get_demo'


class DQfDConfig(Config):
    demo_mode = 'use_demo'
    demo_num = int(Config.BATCH_SIZE * Config.DEMO_RATIO)


