

set -e
set -v

poetry export --no-interaction --format requirements.txt --output requirements.all.txt --with dev
poetry export --no-interaction --format requirements.txt --output requirements.main.txt

set +v
set +e
