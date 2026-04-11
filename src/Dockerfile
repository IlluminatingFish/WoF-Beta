# Base image with CUDA 12.1 and CUDNN for PyTorch 2.4
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies and Miniconda (Python 3.12 comes with Conda env)
RUN apt-get update && \
    apt-get install -y \
        curl \
        wget \
        git \
        build-essential \
        ffmpeg \
        libgl1 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda (latest version supports Python 3.12 in envs)
ENV CONDA_DIR=/opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

ENV GRADIO_SERVER_PORT=7860


# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app


# Create the Conda environment with Python 3.12
RUN conda env create -f environment.yml

# Set CUDA arch list to match NVIDIA L4 (Compute Capability 8.9)
ENV TORCH_CUDA_ARCH_LIST="8.9"

RUN conda run -n ARTalk /bin/bash -c "\
    git clone --recurse-submodules https://github.com/xg-chu/diff-gaussian-rasterization.git && \
    pip install ./diff-gaussian-rasterization && \
    rm -rf ./diff-gaussian-rasterization"

# Run your build script with automatic "yes"
RUN yes | bash ./build_resources.sh


CMD ["conda", "run", "-n", "ARTalk", "python", "inference.py", "--run_app"]