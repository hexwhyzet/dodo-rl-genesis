#!/usr/bin/env bash
set -e

# start virtual display
Xvfb :1 -screen 0 "${RESOLUTION}" -ac +extension GLX +render -noreset &
sleep 1

# start XFCE desktop on virtual display
DISPLAY=:1 startxfce4 &
sleep 2

# start VNC server (no password by default — restrict via firewall/SSH tunnel)
x11vnc -display :1 -forever -nopw -quiet -bg

# start NoMachine if installed
if command -v /etc/NX/nxserver &>/dev/null; then
    /etc/NX/nxserver --startup || true
fi

exec "$@"
