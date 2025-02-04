#!/bin/bash
set -e

sudo apt-get -y update
sudo apt-get -y upgrade
sudo DEBIAN_FRONTEND=noninteractive  apt-get install --no-install-recommends -y \
	htop \
	iproute2 \
	lcov \
	menu \
	mesa-utils \
	novnc \
	libturbojpeg \
	openbox \
	python3-catkin-tools \
	python3-jinja2 \
	python3-numpy \
	python3-websockify \
	python3-xdg \
	python3-xmltodict \
	qt5dxcb-plugin \
	screen \
	terminator \
	vim \
	x11vnc \
	xfce4 \
	xvfb

sudo DEBIAN_FRONTEND=noninteractive  apt-get install --no-install-recommends -y \
	vim \
	gstreamer1.0-plugins-bad \
	gstreamer1.0-libav \
	gstreamer1.0-gl \
	libqt5gui5 \
	bash-completion \
	libfuse3-3 \
	fuse \
	libcanberra-gtk-module \
	libpulse-mainloop-glib0 \
	ca-certificates \
	libgstreamer-plugins-bad1.0-dev \
	libgstreamer-plugins-base1.0-dev \
	libgstreamer-plugins-good1.0-dev \
	python3-future \
	python3-lxml \
	python3-packaging \
	python3-tk \
	python3-toml \
	ros-noetic-fcl \
	ros-noetic-fcl-catkin \
	ros-noetic-octomap-ros \
	ros-noetic-octomap-server \
	ros-noetic-ompl \
	xterm
