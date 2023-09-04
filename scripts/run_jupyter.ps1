# "Separable 2D filters for note onset detection in music signals"
#
#  Copyright (C) 2023 Peter Steiner
# License: BSD 3-Clause

python.exe -m venv venv

.\venv\Scripts\activate.ps1

python.exe -m pip install --editable '.[notebook]'
jupyter notebook

deactivate
