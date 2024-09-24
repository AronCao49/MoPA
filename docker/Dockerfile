FROM nvcr.io/nvidia/pytorch:21.06-py3
# FROM nvcr.io/nvidia/pytorch:22.09-py3

# Set non-interactive mode
ENV DEBIAN_FRONTEND noninteractive

# Install torchsparse
RUN apt-get update && apt-get install -y libsparsehash-dev
# RUN pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

# Install prerequisite packages for PatchworkPP
RUN apt-get install g++ build-essential libeigen3-dev python3-pip python3-dev cmake -y

# Install spareseconv & auxiliary libs
# RUN pip install --upgrade git+https://github.com/facebookresearch/SparseConvNet.git
RUN pip install --timeout 1000  imageio open3d
RUN pip install --timeout 1000  git+https://github.com/facebookresearch/segment-anything.git
