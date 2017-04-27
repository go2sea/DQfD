# DQfD

An implement of DQfD（Deep Q-learning from Demonstrations) raised by DeepMind:Learning from Demonstrations for Real World Reinforcement Learning

It also compared DQdD with Double DQN on the CarPole game, and the comparation shows that DQfD outperforms Double DQN.

## Comparation DQfD with Double DQN

![figure_1](/images/figure_1.png)
![figure_2](/images/figure_2.png)
![figure_3](/images/figure_3.png)

At the end of training, the epsilion used in greedy_action is 0.1, and thats the reason why the curves is not so stable.


## Get the expert demo data

Compared to double DQN, a improvement of DQfD is pre-training. DQfD initially trains solely on the demonstration data before starting any interaction with the environment. This code used a network fine trained by Double DQN to generate the demo data.

You can see the details in function:
```
  get_demo_data()
```

## Get Double DQN scores

For comparation, I first trained an network through Double DQN, witch has the same parameters with the DQfD.
```
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
```

## Get DQfD scores

```
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
```
## Map

Finaly, we can use this function to show the difference between Double DQN and DQfD.
```
    map(dqfd_scores=dqfd_mean_scores, ddqn_scores=ddqn_mean_scores, xlabel='Red: dqfd         Blue: ddqn', ylabel='Scores')
```



