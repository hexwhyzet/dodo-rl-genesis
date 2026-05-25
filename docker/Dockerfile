FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    git \
    curl \
    tmux \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["sleep", "infinity"]
