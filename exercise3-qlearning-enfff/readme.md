# Exercise 3
Exercise 3 - Q-learning for cartpole.

See assignment document for instructions.

## Before you start, this is what you need to know

You can read the report [here](report.pdf)

I've added a parser with a few options to simplify your life, you can check those options by running `python qlearning.py`

which should return

``` bash
usage: qlearning.py [-h] [--task TASK] [--mode MODE] [--eps EPS] [--episodes EPISODES] [--render] [--initial_condition INITIAL_CONDITION] [--path PATH]

options:
  -h, --help            show this help message and exit
  --task TASK, -t TASK  Task to run
  --mode MODE, -m MODE  Allowed values: 'TRAINING', 'TEST'
  --eps EPS, -e EPS     Allowed values: 'const', 'glie'
  --episodes EPISODES   Positive integers only
  --render              Render the environment during training
  --initial_condition INITIAL_CONDITION
                        Initial coefficient to initialize Q(s,a)
  --path PATH           Path to the model to load or save

```

## Task 3.1
We're gonna use the arguments `--mode` to choose wether test or train gthe model, then `-eps` for chosing `const` or `glie`, and finally the path where to save the model.

If you wish to run the already trained models, just run
``` bash
    python qlearning.py --task=3.2 --mode="TRAINING" --eps="const" --path="assets/task3.1/const_q_values.npy"
```
or, with epsilon GLIE:
``` bash
    python qlearning.py --task=3.2 --mode="TRAINING" --eps="glie" --path="assets/task3.1/glie_q_values.npy"
```

Otherwise, if you wish to train them from scratch with epsilon constant run:
``` bash
    python qlearning.py --task=3.1 --mode="TRAINING" --eps="const" --episodes=20000
```

or, with epsilon GLIE:
``` bash
    python qlearning.py --task=3.1 --mode="TRAINING" --eps="glie" --episodes=20000
```
## Task 3.2
Since we just have to plot the heatmap, we're gonna use the models computed in the previous task. If you wish to change the episodes number, you can use the option `--episodes`.

For constant epsilon:
``` bash
    python qlearning.py --task=3.2 --mode="TEST" --eps="const" --episodes=200 --path="assets/task3.1/const_q_values.npy"
```
Otherwise, epsilon glie:
``` bash
    python qlearning.py --task=3.2 --mode="TEST" --eps="glie" --episodes=200 --path="assets/task3.1/glie_q_values.npy"
```

## Task 3.3

For this task we're gonna change the initial condition through the argumment `--initial_condition=INT`

You can run my results for respectively with:
``` bash
    python qlearning.py --task=3.3 --mode="TEST" --initial_condition=50 --path="assets/task3.3/50_q_values.npy"
```

You can train the model with any initial condition as you wish. For example, using `50` for initializingthe `q_grid` tensor, with constant epsilon:
``` bash
    python qlearning.py --task=3.3 --mode="TRAINING" --initial_condition=50 --path="assets/task3.3/50_q_values.npy"
```