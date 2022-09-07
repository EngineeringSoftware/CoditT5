#!/bin/bash

# from https://github.com/EngineeringSoftware/time-segmented-evaluation/blob/main/python/prepare_conda_env.sh

_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# ========== Prepare Conda Environments

DEFAULT_CONDA_PATH="$HOME/opt/anaconda3/etc/profile.d/conda.sh"
PYTORCH_VERSION=1.9.0
TORCHVISION_VERSION=0.10.0

function get_conda_path() {
        local conda_exe=$(which conda)
        if [[ -z ${conda_exe} ]]; then
                echo "Fail to detect conda! Have you installed Anaconda/Miniconda?" 1>&2
                exit 1
        fi

        echo "$(dirname ${conda_exe})/../etc/profile.d/conda.sh"
}

function get_cuda_version() {
        local nvidia_smi_exe=$(which nvidia-smi)
        if [[ -z ${nvidia_smi_exe} ]]; then
                echo "cpu"
        else
                local cuda_version_number="$(nvcc -V | grep "release" | sed -E "s/.*release ([^,]+),.*/\1/")"
                case $cuda_version_number in
                10.2*)
                        echo "cu102";;
                11.3*)
                        echo "cu113";;
                *)
                        echo "Unsupported cuda version $cuda_version_number!" 1>&2
                        exit 1
                esac
        fi
}


function prepare_conda_env() {
        ### Preparing the base environment "tseval"
        local env_name=${1:-cdt}; shift
        local conda_path=${1:-$(get_conda_path)}; shift
        local cuda_version=${1:-$(get_cuda_version)}; shift

        echo ">>> Preparing conda environment \"${env_name}\", for cuda version: ${cuda_version}; conda at ${conda_path}"
        
        # Preparation
        set -e
        set -x
        source ${conda_path}
        conda env remove --name $env_name
        conda create --name $env_name python=3.8 pip -y
        conda activate $env_name

        # PyTorch
        local cuda_toolkit="";
        case $cuda_version in
        cpu)
                cuda_toolkit=cpuonly;;
        cu102)
                cuda_toolkit="cudatoolkit=10.2";;
        cu113)
                cuda_toolkit="cudatoolkit=11.3 -c conda-forge";;
        *)
                echo "Unexpected cuda version $cuda_version!" 1>&2
                exit 1
        esac
        
        conda install -y pytorch=${PYTORCH_VERSION} torchvision=${TORCHVISION_VERSION} ${cuda_toolkit} -c pytorch

        conda install -y -c conda-forge jsonnet

        # Other libraries
        pip install -r requirements.txt
}


prepare_conda_env "$@"
