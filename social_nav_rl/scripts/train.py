import os
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.gridworld_env import CustomGridWorldEnv
from utils.arg_parser import parse_args
from utils.callbacks import EpisodeTrackingCallback
from utils.smooth import gaussian_smooth

def main():
    args = parse_args()
    coords = args.coordinates or [(4.5*33, 3*33), (8.25*33, 9*33), (9*33, 10.5*33)]
    ors = args.orientation or [-90, 90, -90]
    env = CustomGridWorldEnv(start=args.start, end=args.end,
                              n_people=args.n_people, coordinates=coords,
                              orientation=ors)
    os.makedirs(env.save_path, exist_ok=True)
    checkpoint_cb = CheckpointCallback(save_freq=1000, save_path=f"./{args.n_people}people/model")
    log_path = "log/log.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write("Episode,Steps,Reward,Route\n")
    episode_cb = EpisodeTrackingCallback(log_file_path=log_path)

    model = A2C('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[64,128,256,128,64]),
                learning_rate=5e-4, gamma=0.8,
                verbose=1, tensorboard_log="./log", device="cuda")
    model.learn(total_timesteps=args.total_timesteps,
                callback=[checkpoint_cb, episode_cb])

    smoothed = gaussian_smooth(env.route)
    env.route = smoothed
    with open(log_path, "a") as f:
        f.write(f"Final Smooth Route: {env.route}\n")
    env.render_result(os.path.join(env.save_path, "RL.png"))

if __name__ == "__main__":
    main()
