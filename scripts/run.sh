#!/bin/sh

# "Separable 2D filters for note onset detection in music signals"
#
# Copyright (C) 2023 Peter Steiner
# License: BSD 3-Clause

python3 -m venv venv
source venv/bin/activate
python3 -m pip install --editable .
python3 src/main.py --plot --export --serialize

deactivate
