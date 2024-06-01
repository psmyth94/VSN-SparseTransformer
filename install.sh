#!/bin/bash

ENV_PATH=""
ENV_NAME="vsnst"
FORCE_REINSTALL=true

# source conda if not already done
if [ -z "$CONDA_EXE" ]; then
    # check if conda is installed in root
    if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source /opt/conda/etc/profile.d/conda.sh
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source $HOME/miniconda3/etc/profile.d/conda.sh
    else
        echo "Conda not found. Please install conda and re-run this script."
        exit 1
    fi
fi

source activate base

if [ -z "$ENV_PATH" ]; then
    ENV_PATH=$CONDA_PREFIX/envs/$ENV_NAME
else
    ENV_PATH=$ENV_PATH/$ENV_NAME
fi

# Function to check if conda environment exists
env_exists() {
    conda env list | grep -q "$ENV_PATH"
    return $?
}


# Step 1: Update or create conda environment. If force reinstall is set to true, then the environment will be reinstalled

if env_exists; then
    if [ "$FORCE_REINSTALL" = true ]; then
        echo "Removing existing conda environment at $ENV_PATH"
        conda env remove -n $ENV_NAME -y
        conda env create -f environment.yml -p $ENV_PATH
    else
        echo "Updating existing conda environment at $ENV_PATH"
        source activate $ENV_NAME
        conda env update -f environment.yml --prune
    fi
else
    echo "Creating new conda environment at $ENV_PATH"
    conda env create -f environment.yml -p $ENV_PATH
    conda activate $ENV_PATH
fi

