import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor


class Trainer:
    """Thin wrapper around SB3 that wires up checkpointing and logging."""

    def __init__(self, env, cfg: Dict[str, Any]):
        self.env = Monitor(env)
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        run_dir = Path(cfg.get("run_dir", "runs/default"))
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = PPO(
            policy=cfg.get("policy", "MlpPolicy"),
            env=self.env,
            device=self.device,
            tensorboard_log=str(run_dir),
            verbose=1,
            **cfg.get("ppo_kwargs", {}),
        )

        self.callbacks = [
            CheckpointCallback(
                save_freq=cfg.get("save_freq", 10_000),
                save_path=str(checkpoint_dir),
                name_prefix=cfg.get("run_name", "model"),
            ),
        ]

        wandb_key = os.environ.get("WANDB_API_KEY")
        if wandb_key:
            self._setup_wandb(cfg)

    def _setup_wandb(self, cfg: Dict[str, Any]) -> None:
        import wandb
        from wandb.integration.sb3 import WandbCallback

        wandb.init(
            project=cfg.get("project", "dodo-rl-genesis"),
            name=cfg.get("run_name", "run"),
            config=cfg,
            sync_tensorboard=True,
        )
        self.callbacks.append(WandbCallback(verbose=0))

    def train(self, total_timesteps: Optional[int] = None) -> None:
        steps = total_timesteps or self.cfg.get("total_timesteps", 1_000_000)
        self.model.learn(
            total_timesteps=steps,
            callback=self.callbacks,
            reset_num_timesteps=True,
        )

    def load(self, path: str) -> None:
        self.model = PPO.load(path, env=self.env, device=self.device)
