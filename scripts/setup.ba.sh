#!/bin/bash

set -e

target_config=$1
echo "target_config:'${target_config}'"

PIP_CLI_OPTS=""
requirements="requirements.txt"
if [[ "$target_config" == "release" ]]; then
    echo "Installing ONLY prd requirements..."
    requirements="requirements.release.txt"
else
    PIP_CLI_OPTS="-e"
    echo "Installing everything..."
fi

scripts_path="$(dirname "$(readlink -f "$0")")"
echo "Script directory: ${scripts_path}"

set -v


# pip
python -m pip install --verbose --upgrade pip

# Things that need to be up-to-date, like pipx
python -m pip install --verbose --upgrade --requirement requirements.upgrade.txt

# Things that need to be installed via pipx, like poetry
bash ${scripts_path}/pipx/install.ba.sh

# requirements, depending on dev|release
python -m pip install --verbose --requirement ${requirements}

# src
python -m pip install --verbose ${PIP_CLI_OPTS} .

set +v
set +e
