from stable_baselines3.common.callbacks import BaseCallback

class EpisodeTrackingCallback(BaseCallback):
    def __init__(self, log_file_path, verbose=0):
        super().__init__(verbose)
        self.log_file_path = log_file_path

    def _on_step(self) -> bool:
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        if dones and dones[0]:
            info = infos[0]
            ep = info.get("current_episode", -1)
            with open(self.log_file_path, "a") as f:
                f.write(f"Episode:{ep} Steps:{info.get('num_step')} Reward:{info.get('reward')} Route:{info.get('route')}\n")
        return True
