#!/usr/bin/env python3
"""Evaluate a saved checkpoint headlessly."""
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    # TODO: instantiate your env and load model
    # from src.envs.my_task import MyTaskEnv
    # from stable_baselines3 import PPO
    # env = MyTaskEnv(cfg={}, show_viewer=False)
    # model = PPO.load(args.checkpoint, env=env)

    # rewards = []
    # for _ in range(args.episodes):
    #     obs, _ = env.reset()
    #     done, ep_reward = False, 0.0
    #     while not done:
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, r, term, trunc, _ = env.step(action)
    #         ep_reward += r
    #         done = term or trunc
    #     rewards.append(ep_reward)
    # print(f"Mean reward over {args.episodes} episodes: {np.mean(rewards):.2f}")

    print(f"Would evaluate checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
