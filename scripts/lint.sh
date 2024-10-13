#!/bin/bash

python3 -m black teta
python3 -m isort teta
python3 -m pylint teta
python3 -m pydocstyle teta
python3 -m mypy --strict teta