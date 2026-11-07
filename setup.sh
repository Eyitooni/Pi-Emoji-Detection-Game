#!/usr/bin/env bash
# One-time setup for Raspberry Pi 5 + Camera Module 3 (Raspberry Pi OS Bookworm, 64-bit)
set -e
sudo apt update
sudo apt install -y python3-picamera2 python3-pil python3-pil.imagetk python3-tk python3-flask python3-opencv
# --system-site-packages lets the venv see picamera2 (installed by apt)
python3 -m venv --system-site-packages venv
./venv/bin/pip install -r requirements.txt
echo
echo "Done. Test the camera:  rpicam-hello"
echo "Web page:  ./venv/bin/python web_app.py"
echo "Desktop:   ./venv/bin/python gui_app.py"
