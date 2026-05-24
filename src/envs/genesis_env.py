import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple


class GenesisEnv(gym.Env):
    """Gymnasium-compatible wrapper around a Genesis scene.

    Subclass this and implement _build_scene, _get_obs, _compute_reward,
    and _is_done for each specific task.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Dict[str, Any], show_viewer: bool = False):
        super().__init__()
        self.cfg = cfg
        self.show_viewer = show_viewer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._init_genesis()
        self._build_scene()

        self.observation_space = self._make_obs_space()
        self.action_space = self._make_action_space()

    def _init_genesis(self) -> None:
        import genesis as gs

        gs.init(backend=gs.cuda if self.device == "cuda" else gs.cpu)
        self.gs = gs

    def _build_scene(self) -> None:
        """Create and populate the Genesis scene. Override in subclass."""
        self.scene = self.gs.Scene(show_viewer=self.show_viewer)

    def _make_obs_space(self) -> spaces.Space:
        raise NotImplementedError

    def _make_action_space(self) -> spaces.Space:
        raise NotImplementedError

    def _get_obs(self) -> np.ndarray:
        raise NotImplementedError

    def _compute_reward(self) -> float:
        raise NotImplementedError

    def _is_done(self) -> bool:
        raise NotImplementedError

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.scene.reset()
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self._apply_action(action)
        self.scene.step()
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_done()
        return obs, reward, terminated, False, {}

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply action to actuators. Override in subclass."""
        raise NotImplementedError

    def close(self) -> None:
        pass
