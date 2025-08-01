# Socially-Aware Navigation with Reinforcement Learning

This repository contains a reinforcement learning framework for socially-aware navigation in a 2D grid-based environment. The agent learns to navigate toward a goal while avoiding human obstacles, whose positions and orientations are considered in a social energy map.

## 🧠 Key Features

- **Custom GridWorld Environment**: Supports arbitrary agent start/end positions, human positions, and orientations.
- **Social Energy Map**: Derived from human location and facing direction.
- **Reinforcement Learning**: Uses A2C or PPO via Stable-Baselines3.
- **Visualization**: Pygame-based rendering and trajectory visualization.
- **Logging**: Automatically logs episode reward, route, and steps.
- **Trajectory Smoothing**: Optional Gaussian smoothing after training.

## 📁 Project Structure

social_nav_rl/

├── envs/

│ └── gridworld_env.py # Custom RL environment

├── scripts/

│ ├── train.py # Main training script

│ └── test.py # Model testing script

├── utils/

│ ├── callbacks.py # Episode tracking and logging

│ ├── smooth.py # Gaussian smoothing utils

│ └── arg_parser.py # CLI argument parser

├── energy_map/

│ └── normalized_get_eng_map.py

├── log/ # Log files

├── pic/ # Rendered trajectories

├── main.py # Default entry point

├── requirements.txt

└── README.md


## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```


### 2. Train the model

```bash
python scripts/train.py
```

You can customize the training by modifying CLI arguments, such as:
```bash
python scripts/train.py --start "[450, 150]" --end "[150, 450]" --n_people 3
```
 
### 3. Test the model (optional)
```bash
python scripts/test.py
```
## CLI Arguments

| Argument            | Description                          | Default                |
| ------------------- | ------------------------------------ | ---------------------- |
| `--start`           | Agent start position                 | `[13.5*33, 4.5*33]`    |
| `--end`             | Agent goal position                  | `[4.5*33, 13.5*33]`    |
| `--n_people`        | Number of human obstacles            | `3`                    |
| `--coordinates`     | List of human positions              | `[(4.5*33,3*33), ...]` |
| `--orientation`     | List of human orientations (degrees) | `[-90, 90, -90]`       |
| `--total_timesteps` | Number of training timesteps         | `10000`                |

