#!/usr/bin/env bash
# step_1: chmod +x aimet_install_py37.sh
# step_2: ./aimet_install_py37.sh co-lib-py37
# Failed: conda activate base && conda remove -n co-lib-py37 --all && rm -rf ./co_lib && conda env list

if [ $# -lt 1 ]; then 
    echo "Environment name is required"
    exit
fi

env_name=$1  # TODO: env_name="co-lib-py37"

function current_base_path {
    conda env list | grep base | awk '{print $NF}'
}

function current_env_path {
    conda env list | grep $1 | awk '{print $NF}'
}

function show_aimet_path {
    echo "$(pip list | grep -i aimet | grep -i common | awk '/\s+/ {print $1}' | xargs pip show | grep -i location | awk '/\s+/ {print $2}')"
}

function test_env {
    python -c "import co_lib; print('successfully instal aimet and co_lib')"
}

function assert {
    if [ $? -ne 0 ]; then
        echo "Command Failed."
        exit 1
    fi
}

# NVIDIA checking
nvidia-smi
assert
conda -h > /dev/null
assert

# co-lib download
app_pwd=fLA3kM6sr3VJJkQtmT6J
co_lib_repo_url="https://cocopie-ai-dongli:$app_pwd@bitbucket.org/cocopie/co_lib.git"
git clone $co_lib_repo_url
assert

# conda_path
conda_path="$(current_base_path)"  # TODO: conda env list | grep base
source ${conda_path}/etc/profile.d/conda.sh
conda create -n $env_name python=3.7 -y
conda activate $env_name
python -V

# TODO: conda & pip install
conda install -c conda-forge blas -y  # sudo find / -name liblapacke.so.3
python -m pip install pip --upgrade

# Aimet version and installing
export AIMET_VARIANT="torch_gpu"
export release_tag="1.19.1.py37"  # TODO: please check version in https://github.com/quic/aimet/releases
export download_url="https://github.com/quic/aimet/releases/download/${release_tag}"
export wheel_file_suffix="cp37-cp37m-linux_x86_64.whl"
pip --default-timeout=600 install ${download_url}/AimetCommon-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}
pip --default-timeout=600 install ${download_url}/AimetTorch-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix} -f https://download.pytorch.org/whl/torch_stable.html  # why pip uninstall torch
pip --default-timeout=600 install ${download_url}/Aimet-${AIMET_VARIANT}_${release_tag}-${wheel_file_suffix}

# Aimet path
aimet_common_path="$(show_aimet_path)/aimet_common"
env_path="$(current_env_path $env_name)"
echo -e "*************************************************************************************************************************************"
echo -e "aimet_common_path=\"$aimet_common_path\""
echo -e "env_path=\"$env_path\""
echo -e "conda_path=\"$conda_path\""
echo -e "*************************************************************************************************************************************\n\n\n"
source "$aimet_common_path/bin/envsetup.sh"
export LD_LIBRARY_PATH="$aimet_common_path/x86_64-linux-gnu:$env_path/lib:$aimet_common_path:$aimet_common_path/lib:$conda_path/lib":$LD_LIBRARY_PATH
export PYTHONPATH="$aimet_common_path":"$aimet_common_path/x86_64-linux-gnu":$PYTHONPATH

# co-lib ENV test
pip install protobuf==3.20.0  # would uninstall higher protobuf
pip install dotmap==1.3.30
pip install scikit-image==0.16.1
cd co_lib
python setup.py install
test_env
