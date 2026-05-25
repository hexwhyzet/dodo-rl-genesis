#!/usr/bin/env bash
set -e

# virtual display for Genesis headless rendering
Xvfb :99 -screen 0 1024x768x24 -ac &

# setup SSH
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A
service ssh start

exec sleep infinity
