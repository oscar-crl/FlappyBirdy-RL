import os
import sys
import time
from tkinter import *

import os

from stable_baselines3 import PPO

from CrappyBirdEnv import CrappyBirdEnv
from GameInfo import GameInfo
from GameWindow import GameWindow

AI_ON = True
AI_FALSE = False
globals()['jump'] = False
GAME_MODE = False


def update(root, window):
    window.tick(globals()['jump'])
    globals()['jump'] = False
    root.after(int(1000 / 60), lambda: update(root, window))


def enable_jump():
    globals()['jump'] = True


def render_test(model, env):
    for episode in range(1000):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            time.sleep(1 / 120)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            score += reward
            print(obs)
            print(action)
            print(score)
            print('')


def init():
    root = Tk()

    root.title("Flappy Birdy")
    GameInfo.init_info(root, AI_ON)

    game_window = GameWindow(root)

    return root, game_window


def game():
    root, game_window = init()
    root.bind('<Key>', lambda e: enable_jump())

    update(root, game_window)
    while True:
        root.update_idletasks()
        root.update()


def process():
    root, game_window = init()

    env = CrappyBirdEnv(root, game_window)

    model = PPO('MlpPolicy', env, verbose=1)

    model.learn(total_timesteps=300000)

    PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_flappy_1')
    model.save(PPO_Path)
    # model.load(PPO_Path, env=env)

    render_test(model, env)


process()
