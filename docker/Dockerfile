FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISPLAY=:99

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    openssh-server \
    git \
    curl \
    tmux \
    vim \
    make \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl -fsSL https://claude.ai/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

RUN mkdir /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

COPY docker/tmux.conf /root/.tmux.conf
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

EXPOSE 22 8080 6006

ENTRYPOINT ["/entrypoint.sh"]
