import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train/test socially-aware navigation RL")
    parser.add_argument("--start", type=float, nargs=2, default=[13.5*33, 4.5*33])
    parser.add_argument("--end", type=float, nargs=2, default=[4.5*33, 13.5*33])
    parser.add_argument("--n_people", type=int, default=3)
    parser.add_argument("--coordinates", type=float, nargs="*", default=[])
    parser.add_argument("--orientation", type=float, nargs="*", default=[])
    parser.add_argument("--total_timesteps", type=int, default=10000)
    return parser.parse_args()
