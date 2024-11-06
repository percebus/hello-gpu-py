#!/bin/bash

set -e
set -x

pip freeze > requirements.frozen.txt

set +x
set +e
