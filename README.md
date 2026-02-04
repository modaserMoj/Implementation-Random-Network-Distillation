**Status:** Archive (code is provided as-is, no updates expected)

## [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) ##


Yuri Burda*, Harri Edwards*, Amos Storkey, Oleg Klimov<br/>
&#42;equal contribution

OpenAI<br/>
University of Edinburgh


### Installation and Setup

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

### What to Expect
- The agent will start exploring and discovering rooms driven by curiosity (RND intrinsic rewards)
- Episode rewards (`eprew`) will be 0 initially - this is normal for hard exploration games
- Watch `n_rooms` to see exploration progress
- Actual game scores typically appear after millions of timesteps
- The paper achieved 8,152 score on Montezuma's Revenge after extensive training

### [Blog post and videos](https://blog.openai.com/reinforcement-learning-with-prediction-based-rewards/)
