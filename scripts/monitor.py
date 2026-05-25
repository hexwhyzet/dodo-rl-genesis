#!/usr/bin/env python3
"""Standalone web monitor server. Run alongside training, stays up indefinitely.

Usage:
    uv run python scripts/monitor.py --log-dir runs/dodo-balance --port 8080
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.streaming import server as stream_server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    video_dir = Path(args.log_dir) / "videos"
    stream_server.start(video_dir=video_dir, port=args.port)
    print(f"[monitor] http://0.0.0.0:{args.port}  (video_dir: {video_dir})")

    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
