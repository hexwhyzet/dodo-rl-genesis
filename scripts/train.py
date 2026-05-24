#!/usr/bin/env python3
"""Entry point for training runs."""
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # TODO: replace with your concrete env
    # from src.envs.my_task import MyTaskEnv
    # env = MyTaskEnv(cfg=OmegaConf.to_container(cfg.env), show_viewer=cfg.env.show_viewer)

    # from src.trainers import Trainer
    # trainer = Trainer(env=env, cfg=OmegaConf.to_container(cfg.trainer))
    # trainer.train()


if __name__ == "__main__":
    main()
