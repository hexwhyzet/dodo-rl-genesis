"""Records training video by capturing frames during the normal training loop."""
from pathlib import Path

SUFFIXES = ["_front", "_side"]


class VideoRecorder:
    def __init__(self, env, video_dir: Path, every_n_iter: int = 50, fps: int = 50):
        self.env = env
        self.video_dir = video_dir
        self.every_n_iter = every_n_iter
        self.fps = fps
        self._recording = False
        video_dir.mkdir(parents=True, exist_ok=True)

    def on_iteration_start(self, iteration: int) -> None:
        if iteration % self.every_n_iter == 0:
            self._paths = [
                self.video_dir / f"iter_{iteration:06d}{suf}.mp4"
                for suf in SUFFIXES
            ]
            for cam in self.env.render_cameras:
                cam.start_recording()
            self._recording = True

    def on_step(self) -> None:
        if self._recording:
            for cam in self.env.render_cameras:
                cam.render()

    def on_iteration_end(self, iteration: int) -> None:
        if self._recording and (iteration + 1) % self.every_n_iter == 0:
            for cam, path in zip(self.env.render_cameras, self._paths):
                cam.stop_recording(save_to_filename=str(path), fps=self.fps)
            self._recording = False
