# Start from Ubuntu 20.04
FROM ubuntu:20.04

# Set noninteractive installation (no pop-ups)
ENV DEBIAN_FRONTEND=noninteractive

# Install basic dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    scons \
    m4 \
    zlib1g-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-perftools-dev \
    libboost-all-dev \
    git \
    wget \
    unzip \
    vim \
    sudo \
    ca-certificates

# Install some Python packages (optional)
RUN pip3 install --upgrade pip
RUN pip3 install matplotlib pandas

# Clone gem5
WORKDIR /opt
RUN git clone https://gem5.googlesource.com/public/gem5
WORKDIR /opt/gem5

# Optionally checkout a stable release (recommended)
# RUN git checkout v23.0

# Build gem5 (example for X86 FS)
RUN scons build/X86/gem5.opt -j$(nproc)

# Set working directory
WORKDIR /opt/gem5

# Default command
CMD ["/bin/bash"]
