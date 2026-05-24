FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DISPLAY=:1 \
    RESOLUTION=2560x1440x24

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    tmux \
    # OpenGL / display
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglu1-mesa \
    # virtual display + VNC
    xvfb \
    x11vnc \
    xfce4 \
    xfce4-terminal \
    dbus-x11 \
    # NoMachine deps
    libgtk-3-0 \
    libpulse0 \
    libnss3 \
    && rm -rf /var/lib/apt/lists/*

# Install NoMachine
RUN wget -q https://download.nomachine.com/download/8.14/Linux/nomachine_8.14.2_1_amd64.deb -O /tmp/nomachine.deb \
    && dpkg -i /tmp/nomachine.deb || true \
    && rm /tmp/nomachine.deb

# Install Claude Code (standalone, no Node.js required)
RUN curl -fsSL https://claude.ai/install.sh | sh \
    && echo 'export PATH="$HOME/.claude/bin:$PATH"' >> /root/.bashrc

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 4000 5900

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
