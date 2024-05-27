FROM nvcr.io/nvidia/pytorch:21.06-py3
# FROM nvcr.io/nvidia/pytorch:22.09-py3

# Set non-interactive mode
ENV DEBIAN_FRONTEND noninteractive

# Install torchsparse
RUN apt-get update && apt-get install -y libsparsehash-dev
# RUN pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# Install Open3d for PatchworkPP
RUN pip install --timeout 1000  open3d 
RUN apt-get install -y libeigen3-dev
RUN git clone https://github.com/isl-org/Open3D
RUN apt-get update && apt-get install -y \
    xorg-dev \
    libxcb-shm0 \
    libglu1-mesa-dev \
    python3-dev \
    # Filament build-from-source
    clang \
    libc++-dev \
    libc++abi-dev \
    libsdl2-dev \
    ninja-build \
    libxi-dev \
    # ML
    libtbb-dev \
    # Headless rendering
    libosmesa6-dev \
    # RealSense
    libudev-dev \
    autoconf \
    libtool 
# Install command (can change the core number for different setup)
RUN cd Open3D && mkdir build && cd build && \
    cmake .. && make -j4 && make install

# Install spareseconv & auxiliary libs
# RUN pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
RUN pip install --timeout 1000  imageio
RUN pip install --timeout 1000  git+https://github.com/facebookresearch/segment-anything.git
