# Robot Learning Laboratories
Labs done for the [Robot Learning](https://didattica.polito.it/pls/portal30/gap.pkg_guide.viewGap?p_cod_ins=01HFNOV&p_a_acc=2024&p_header=S&p_lang=IT&multi=N) class. I wrote a report for each lab discussing the main intuitions, issues and the results.

Brief description of the labs, and links to the sources:
1. [**Implementation of the Extended Kalman Filter in ROS**](exercise1-ekf-enfff/Exercise1.pdf). Read the [Report](exercise1-ekf-enfff/report.pdf).
2. [**Reinforcement Learning fundamentals**](exercise2a-rl-fundamentals-enfff/Exercise2a.pdf): implementation of the Linear Quadratic Regulator (LQR), a standard control strategy, to control [Gym's Cartpole environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) and then comparing the results with a basic Reinforcement Learning algorithm, seen as a black-box, where I designed and tested different reward functions. Read the [Report](exercise2a-rl-fundamentals-enfff/report.pdf)
3. [**Q-learning**](exercise3-qlearning-enfff/Exercise3.pdf): implementation of a tabular Q-Learning method to control to control [Gym's Cartpole environment](https://www.gymlibrary.dev/environments/classic_control/cart_pole/). Read the [Report](exercise3-qlearning-enfff/report.pdf).
4. [**Policy Gradient Algorithms**](exercise4-policygradient-enfff/Exercise4.pdf): study of the REINFORCE algorithm, its variant with the baseline, and two Actor-Critic methods: [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html), [PPO](https://openai.com/research/openai-baselines-ppo). Read the [Report](exercise4-policygradient-enfff/report.pdf).