# DQfD

An implementation of DQfD（Deep Q-learning from Demonstrations) raised by DeepMind:Learning from Demonstrations for Real World Reinforcement Learning

It also compared DQfD with Double DQN on CartPole game, and the comparison shows that DQfD outperforms Double DQN.

Algorithm is different between Version3(DQfD.py) and Version-1(DQfDDDQN.py). Compared to V1, the V3 DQfD added:
 
```
    prioritized replay
    n-step TD loss
    importance sampling
```


## Comparison between DQfD(V3) and Double DQN

Compared to V1, the V3 DQfD added prioritized replay, n-step TD loss, importance sampling.

![figure_0](/images/dqfd-v3_vs_ddqn.png)

Note: In my experiments on CartPole, the n-step TD loss is Counterproductive and leads to worse performance. And the parameter λ for loss_n_step for the result you see above is 0. Maybe I implement n-step TD loss in wrong way and I hope someone can explain that.

## Comparison between DQfD(V1) and Double DQN

![figure_1](/images/figure_1.png)
![figure_2](/images/figure_2.png)
![figure_3](/images/figure_3.png)

At the end of training, the epsilon used in greedy_action is 0.01.


## Get the expert demo transitions

Compared to double DQN, a improvement of DQfD is pre-training. DQfD initially trains solely on the demonstration data before starting any interaction with the environment. This code used a network fine trained by Double DQN to generate the demo data.

You can see the details in function:
```
  get_demo_data()
```

## Get Double DQN scores

For Comparison, I first trained an network through Double DQN, witch has the same net with the DQfD.
```
    # --------------------------  get DDQN scores ----------------------------------
    ddqn_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DDQN(i, env)
        ddqn_sum_scores = np.array([a + b for a, b in zip(scores, ddqn_sum_scores)])
    ddqn_mean_scores = ddqn_sum_scores / Config.iteration
```

## Get DQfD scores

```
    # ----------------------------- get DQfD scores --------------------------------
    dqfd_sum_scores = np.zeros(Config.episode)
    for i in range(Config.iteration):
        scores = run_DQfD(i, env)
        dqfd_sum_scores = np.array([a + b for a, b in zip(scores, dqfd_sum_scores)])
    dqfd_mean_scores = dqfd_sum_scores / Config.iteration
```
## Map

Finally, we can use this function to show the difference between Double DQN and DQfD.
```
    map(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores, xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
```



