**Status:** Archive (code is provided as-is, no updates expected)

## [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)

Yuri Burda*, Harri Edwards*, Amos Storkey, Oleg Klimov<br/>
\*equal contribution

OpenAI<br/>
University of Edinburgh

### Installation and Setup

**Tested with Python 3.10**

#### 1. Create Virtual Environment

```bash
python -m venv env
```

#### 2. Activate Virtual Environment

**Windows:**

```bash
env\Scripts\activate
```

**Linux/Mac:**

```bash
source env/bin/activate
```

#### 3. Install Dependencies

**Option A: Using requirements.txt**

```bash
pip install -r requirements.txt
pip install autorom[accept-rom-license]==0.6.1
pip install git+https://github.com/openai/baselines.git@master
```

**Option B: Manual installation**

```bash
# Core ML libraries
pip install tensorflow==2.18.0
pip install gym==0.26.2
pip install opencv-python==4.10.0.84
pip install mpi4py==4.0.1

# Atari support
pip install ale-py==0.8.1
pip install autorom[accept-rom-license]==0.6.1

# OpenAI Baselines (legacy version compatible with gym 0.26)
pip install git+https://github.com/openai/baselines.git@master
```

**Note:** This code requires TensorFlow 1.x API compatibility mode which is enabled automatically in the code via `tensorflow.compat.v1`.

### Usage

#### Train on Montezuma's Revenge (default)

```bash
python run_atari.py --gamma_ext 0.999
```

#### Train with Custom Parameters

```bash
# Quick test (100K timesteps, ~10 minutes)
python run_atari.py --gamma_ext 0.999 --num-timesteps 100000

# Short training (10M timesteps, ~12 hours)
python run_atari.py --gamma_ext 0.999 --num-timesteps 10000000

# Different game
python run_atari.py --env PongNoFrameskip-v4 --gamma_ext 0.999 --num-timesteps 10000000
```

#### Multi-GPU/Multi-Machine Training

To use more than one gpu/machine, use MPI:

```bash
# 8 GPUs with 128 environments per GPU (1024 total)
mpiexec -n 8 python run_atari.py --num_env 128 --gamma_ext 0.999
```

### Key Arguments

- `--num-timesteps`: Total timesteps to train (default: 1 trillion, effectively infinite)
- `--num_env`: Number of parallel environments (default: 32)
- `--gamma_ext`: Discount factor for extrinsic rewards (default: 0.99, paper uses 0.999)
- `--env`: Environment name (default: MontezumaRevengeNoFrameskip-v4)
- `--proportion_of_exp_used_for_predictor_update`: Fraction of experience for RND predictor training (default: 1.0, paper uses 0.25)
- `--goal_weight`: Enable goal-oriented weighted intrinsic reward modification (default: 0, set to 1 to enable)

### Example Commands

```bash
# Baseline RND on Montezuma's Revenge (15M steps)
python run_atari.py --env MontezumaRevengeNoFrameskip-v4 --num_env 64 --gamma_ext 0.999 --num-timesteps 15000000

# Baseline with paper's predictor proportion setting
python run_atari.py --env MontezumaRevengeNoFrameskip-v4 --num_env 64 --gamma_ext 0.999 --proportion_of_exp_used_for_predictor_update 0.25 --num-timesteps 15000000

# With goal-weight modification enabled
python run_atari.py --env MontezumaRevengeNoFrameskip-v4 --num_env 64 --gamma_ext 0.999 --proportion_of_exp_used_for_predictor_update 0.25 --goal_weight 1 --num-timesteps 15000000

# Venture environment
python run_atari.py --env VentureNoFrameskip-v4 --num_env 64 --gamma_ext 0.999 --num-timesteps 15000000
```

### Goal-Weight Modification

This fork includes an experimental modification: **Goal-Oriented Weighted Intrinsic Reward**.

The intrinsic reward is weighted by the sigmoid of the extrinsic value estimate:

```
i_t' = i_t * σ(V_ext(s_t))
```

With a 4M step warmup period (pure RND exploration first), then goal-weight activates.

Enable with `--goal_weight 1`.

### [Blog post and videos](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/)
