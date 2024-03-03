# Exercise4
Exercise 4 - RL Policy Gradient algorithms

For this exercise you must use gym v0.21.0 and stable-baselines3 v1.6.2.

You can install the required python packages with this command:
```
pip install -r ./required_python_libs.txt
```
<!-- bold of you to assume it would work on my esoterical linux distribution -->


# What you need to know

I've added an option `--task` to the parser, its allowed arguments are `1a|1b|1c`

To implement the formulas as shown in the [report](report.pdf), I've changed the optimizer initialization, adding two arguments: `maximize=True` and `weight_decay=self.gamma` ([docs](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)), to correctly implement the pseudo-code from the [book, page 352](http://incompleteideas.net/book/RLbook2020.pdf)

# Task 1
You can train the model with
`python cartpole.py --task="1a"`.

This command will train the model for the first ask and save it as `model_ContinuousCartPole-v0_0.mdl`. Now you can test the one you generated, or test mine

`python cartpole.py --test="models/1a.mdl"`

If you're interest in the other points replace the task number with the one you're interested in
`python cartpole.py --test="models/1b.mdl"`
`python cartpole.py --test="models/1c.mdl"`

python cartpole_sb3.py --algo="sac" --test="models/sac_cartpole_100k.zip" --test_episodes=400

<!-- 1a
Average test reward: 52.87 episode length: 52.87
Average test reward: 51.28 episode length: 51.28
Average test reward: 50.83 episode length: 50.83

1b
Average test reward: 67.24 episode length: 67.24
Average test reward: 65.0 episode length: 65.0
Average test reward: 73.87 episode length: 73.87

1c
Average test reward: 35.56 episode length: 35.56
Average test reward: 38.2 episode length: 38.2
Average test reward: 35.88 episode length: 35.88 -->
# Task 4

Train your models with (`algo="ppo"|"sac"`)

`python cartpole_sb3.py --algo="ppo" --total_timesteps=250000`

this will generate a model, `ppo_cartpole.zip`, which you can test with:

`python cartpole_sb3.py --algo="ppo" --test="ppo_cartpole.zip" `

I've trained a few the models, all located in the `models/` folder, feel free to use them. One example:

` python cartpole_sb3.py --algo="ppo" --test="models/sac_cartpole_100k.zip"`

<!-- sac 100k lr 0.003
Test reward (avg +/- std): (434.28 +/- 52.99303350441452) - Num episodes: 100

sac standard lr 0.003
Test reward (avg +/- std): (430.85 +/- 47.79380189940951) - Num episodes: 100

sac lr 0.03 time 661
Test reward (avg +/- std): (394.14 +/- 84.71009621054624) - Num episodes: 100

sac lr 0.3 time 784
Test reward (avg +/- std): (336.0 +/- 72.86219870412916) - Num episodes: 100



0.003 & ? & 430.85 & 47.79
0.03 & 661 & 394.14 & 84.71
0.3 & 784 & 336.0 & 72.86


ppo standard time 37
Test reward (avg +/- std): (167.89 +/- 64.66233756987138) - Num episodes: 100
Test reward (avg +/- std): (163.28 +/- 64.19938940519606) - Num episodes: 100

ppo lr 0.03  - time 47
Test reward (avg +/- std): (165.9 +/- 50.458398706260986) - Num episodes: 100
Test reward (avg +/- std): (162.48 +/- 47.480623416294776) - Num episodes: 100

ppo lr 0.3 - time 36
Test reward (avg +/- std): (309.24 +/- 101.72719597039918) - Num episodes: 100
Test reward (avg +/- std): (278.77 +/- 82.6057933803677) - Num episodes: 100 -->