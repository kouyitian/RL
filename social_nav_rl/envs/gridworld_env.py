import os
import time
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from energy_map.normalized_get_eng_map import get_map

grid_size = 33 * 15  # 定义全局格子尺寸


class CustomGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, start, end, n_people, coordinates, orientation, save_path="pic", render_mode='human', sigma=1):
        super().__init__()
        self.grid_size = grid_size
        self.start = np.array(start, dtype=float)
        self.target_position = np.array(end, dtype=float)
        self.n_people = n_people
        self.obstacles = [np.array(c, dtype=float) for c in coordinates]
        self.orientation = orientation
        self.save_path = save_path
        self.render_mode = render_mode
        self.sigma = sigma
        self.step_length = 15
        self.max_step = 1000

        self.action_space = spaces.Discrete(20)
        self.observation_space = spaces.Box(
            low=np.zeros(4), high=np.array([self.grid_size] * 2 + [2 * self.grid_size] * 2), dtype=np.float32)

        self.current_step = 0
        self.agent_position = self.start.copy()
        self.route = []
        self.total_reward = 0
        self.current_episode = 0

        self.energy_map = self._build_energy_map()
        self.max_eng = self.energy_map.max()

        if self.render_mode == "human":
            self._render_init()

    def _build_energy_map(self):
        base = np.zeros((self.grid_size, self.grid_size))
        for pos, ori in zip(self.obstacles, self.orientation):
            base += get_map(self.grid_size, self.grid_size, pos, ori, self.target_position)
        return np.array(base)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_episode += 1
        self.current_step = 0
        self.total_reward = 0
        self.route = []
        self.agent_position = self.start.copy()
        if self.render_mode == "human":
            self._render_init()
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        delta = self.target_position - self.agent_position
        obs = np.array([*self.agent_position, delta[0], delta[1]], dtype=np.float32)
        return obs

    def _get_info(self):
        return {"distance": np.linalg.norm(self.agent_position - self.target_position)}

    def step(self, action):
        assert self.action_space.contains(action)
        self.current_step += 1
        prev_dist = np.linalg.norm(self.agent_position - self.target_position)

        radians = 2 * np.pi / self.action_space.n * action
        self.agent_position += self.step_length * np.array([np.cos(radians), np.sin(radians)])
        self.agent_position = np.clip(self.agent_position, 0, self.grid_size - 1)
        self.route.append(self.agent_position.tolist())

        if self.render_mode == "human":
            self._render_update()

        new_dist = np.linalg.norm(self.agent_position - self.target_position)
        done = False
        reward = 0

        if new_dist < self.step_length + 10:
            pygame.image.save(self.window, os.path.join(self.save_path, f'Episode{self.current_episode}.png'))
            reward = 500
            done = True
            info = {"is_success": True, "current_episode": self.current_episode,
                    "reward": self.total_reward + reward, "route": self.route, "num_step": self.current_step}
        else:
            energy = self.energy_map[int(self.agent_position[0]), int(self.agent_position[1])]
            reward = (prev_dist - new_dist) / self.step_length - self.sigma * energy - 1
            self.total_reward += reward
            info = {"is_success": False}

        if self.current_step >= self.max_step:
            reward -= 500
            done = True
            info.update({"is_success": False})

        return self._get_observation(), reward, done, False, info

    def _render_init(self):
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((800, 800))
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface(self.window.get_size())
        self.pix_square_size = 800 / self.grid_size
        self._draw_full()

    def _render_update(self):
        self._draw_full()
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def _draw_full(self):
        self.canvas.fill((255, 255, 255))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                val = self.energy_map[x, y]
                c = int((val / self.max_eng) * 255)
                color = (255, 255 - c, 255 - c)
                rect = pygame.Rect(x * self.pix_square_size, y * self.pix_square_size,
                                   self.pix_square_size, self.pix_square_size)
                pygame.draw.rect(self.canvas, color, rect)
        for pos, ori in zip(self.obstacles, self.orientation):
            # 三角或箭头表示障碍方向略
            pass
        # agent & target drawing...
        self.window.blit(self.canvas, (0, 0))

    def render_result(self, filename):
        time.sleep(1)
        pygame.image.save(self.window, filename)
