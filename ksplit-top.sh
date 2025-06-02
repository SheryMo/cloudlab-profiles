# #!/bin/bash 

# VERBOSE_LOG=${HOME}/ksplit-verbose.log

# echo "Logging to ${VERBOSE_LOG}"
# /local/repository/ksplit-setup.sh |& tee -a ${VERBOSE_LOG}
#!/bin/bash

sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
REPO_URL="https://github.com/SheryMo/MoE_Update.git"
REPO_PATH="/local/repository/MoE_Update"
HOME=/local/home/rrMo
export HOME
mkdir -p $HOME

# only on first boot
if ! command -v nvcc &> /dev/null
then

  if [ $(lsb_release -c -s) = "focal" ]
  then 

    sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
    sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
	
    sudo apt update
    sudo apt -y install python3-pip python3-dev
    sudo pip3 install --upgrade pip

    sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb -O /tmp/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring_1.0-1_all.deb
    sudo apt update
    sudo apt-get clean
    sudo apt -y install linux-headers-$(uname -r)
    sudo apt-mark hold cuda-toolkit-12-config-common nvidia-driver-535 # don't allow these
    sudo apt -y install nvidia-driver-520
    sudo apt-get clean
    sudo apt -y install cuda-11-8 cuda-runtime-11-8 cuda-drivers=520.61.05-1
    sudo apt-get clean
    sudo apt -y install nvidia-gds-11-8
    sudo apt-get clean
    sudo apt -y install libcudnn8=8.9.3.28-1+cuda11.8 cuda-toolkit-11-8
    sudo apt-get clean

    echo 'PATH="/usr/local/cuda-11.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"' | sudo tee /etc/environment
    
    sudo chown $USER /data
    mkdir -p /data/tmp
    sudo chown $USER -R /data
    
    wget -c https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O $HOME/miniconda.sh
    bash $HOME/miniconda.sh -b -p $HOME/anaconda3
    export PATH=$HOME/anaconda3/bin:$PATH
    eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
    conda init
    conda create -y -n silly_env python=3.10
    conda activate silly_env
    # conda install -y cudatoolkit
    TMPDIR=/data/tmp pip install --cache-dir=/data/tmp --target=/data datasets==2.16.0 flask huggingface_hub netifaces seaborn
    TMPDIR=/data/tmp pip install --cache-dir=/data/tmp --target=/data -e .

    # TMPDIR=/data/tmp python3 -m pip install --cache-dir=/data/tmp --target=/data Cython==0.29.32
    # TMPDIR=/data/tmp python3 -m pip install --cache-dir=/data/tmp --target=/data -r /local/repository/requirements_cloudlab_dl.txt --extra-index-url https://download.pytorch.org/whl/cu113 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    # TMPDIR=/data/tmp python3 -m pip install --cache-dir=/data/tmp --target=/data jupyter-core jupyter-client jupyter_http_over_ws traitlets -U --force-reinstall
  fi

  if [ $(lsb_release -c -s) = "jammy" ]
  then

    sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
    sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
	
    sudo apt update
    sudo apt -y install python3-pip python3-dev
    sudo pip3 install --upgrade pip

    sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb -O /tmp/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i /tmp/cuda-keyring_1.0-1_all.deb
    sudo apt update
    sudo apt-get clean
    sudo apt -y install linux-headers-$(uname -r)
    sudo apt -y install nvidia-driver-545
    sudo apt-get clean
    sudo apt -y install cuda-12-2 cuda-runtime-12-2 cuda-drivers=545.23.08-1
    sudo apt-get clean
    sudo apt -y install nvidia-gds-12-2
    sudo apt-get clean
    sudo apt -y install libcudnn8=8.9.6.50-1+cuda12.2 cuda-toolkit-12-2
    sudo apt-get clean
    
    wget -c https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh -O $HOME/miniconda.sh
    bash $HOME/miniconda.sh -b -p $HOME/anaconda3
    export PATH=$HOME/anaconda3/bin:$PATH
    eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
    conda init
    conda create -y -n silly_env python=3.10
    conda activate silly_env
    conda install -y cudatoolkit
    TMPDIR=/data/tmp pip install --cache-dir=/data/tmp --target=/data datasets==2.16.0 flask huggingface_hub netifaces seaborn
    TMPDIR=/data/tmp pip install --cache-dir=/data/tmp --target=/data -e .
    
    echo 'PATH="/usr/local/cuda-12.2/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin"' | sudo tee /etc/environment
    
  fi
fi

export PATH=$HOME/anaconda3/bin:$PATH
eval "$($HOME/anaconda3/bin/conda shell.bash hook)"
conda activate silly_env
# 每次都执行：clone repo 并下载模型
rm -rf $REPO_PATH
git clone $REPO_URL $REPO_PATH
mkdir -p $REPO_PATH/ll_test
cd $REPO_PATH/ll_test
python3 <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="llama-moe/LLaMA-MoE-v1-3_5B-2_8",
    local_dir="./",
    local_dir_use_symlinks=False
)
EOF
mkdir -p $REPO_PATH/llama-moe/LLaMA-MoE-v1-3_5B-2_8
mv pytorch_model-00001-of-00002.bin $REPO_PATH/llama-moe/LLaMA-MoE-v1-3_5B-2_8/
mv pytorch_model-00002-of-00002.bin $REPO_PATH/llama-moe/LLaMA-MoE-v1-3_5B-2_8/
mv tokenizer.model $REPO_PATH/llama-moe/LLaMA-MoE-v1-3_5B-2_8/
