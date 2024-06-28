#!/usr/bin/env bash

# Exit script on any error
set -e

# Update package list and install ZBar dependencies
apt-get update && apt-get install -y \
    zbar-tools \
    libzbar-dev

# Clean up to reduce the image size
apt-get clean
rm -rf /var/lib/apt/lists/*
