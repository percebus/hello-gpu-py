#!/bin/bash

set -e
set -v

pip freeze > requirements.frozen.txt

set +v
set +e
