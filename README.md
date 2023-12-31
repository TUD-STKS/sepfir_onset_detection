# Separable 2D filters for note onset detection in music signals
## Metadata
- Author: [Peter Steiner](mailto:peter.steiner@tu-dresden.de),
  [Kilian Göller](mailto:kilian.goeller@tu-dresden.de), 
  and [Peter Birkholz](mailto:peter.birkholz@tu-dresden.de)
- Conference: 2024 IEEE International Conference on Acoustics, Speech and Signal 
  Processing (submitted)
- Weblink: [https://2024.ieeeicassp.org/](https://2024.ieeeicassp.org/)

## Summary and Contents
This repository includes the code accompanying the research paper entitled 
"Separable 2D filters for note onset detection in music signals" to reproduce the results
from the paper. It presents a novel approach to note onset detection in music signals
using separable 2D filters applied to mel spectrograms. First it is shown that linear 
filters achieve equivalent results to the SuperFlux algorithm, which uses nonlinear
filters. Furthermore, the filter kernels are optimized using Particle Swarm Optimization
(PSO).

All experiments can either be reproduced in a Python script that can be run locally or on
a high-performance cluster.


- File list:
    - The file list contains all files provided in the repository together with a 
    short description.
- Usage:
    - How can users get started with your research code. This contains setting up a 
    installing packages in a virtual environment `venv` and running one `main.py` that
    includes your main code. 
    - Very important and often forgotten: How can the data to reproduce the results be
    obtained?
- Acknowledgments:
    - Any kind of required funding information 
    - other acknowledgments, such as project partners, contributors, family etc.
- Referencing:
    - How can your work be cited? Ideally, provide a bibtex entry of the paper.

## File list
- The following scripts are provided in this repository
    - `scripts/run.sh`: UNIX Bash script to reproduce the Figures in the paper.
    - `scripts/run_jupyter.sh`: UNIX Bash script to start the Jupyter Notebook for the
    paper.
    - `scripts/run.bat`: Windows batch script to reproduce the Figures in the paper.
    - `scripts/run_jupyter.ps1`: Windows batch script to start the Jupyter Notebook for
    the paper.
- The following python code is provided in `src`
    - `src/file_handling.py`: Utility functions for storing and loading data and models.
    - `src/preprocessing.py`: Utility functions for preprocessing the dataset
    - `src/main.py`: The main script to reproduce all results.
- `requirements.txt`: Text file containing all required Python modules to be installed
if we are working in Python. It can be obtained by typing 
`pip freeze > requirements.txt` in a PowerShell or in a Bash. 
- `README.md`: The README displayed here.
- `LICENSE`: Textfile containing the license for this source code. You can find 
- `data/`: The optional directory `data` contains
    - `train.csv`: Training data as CSV file
    - `test.csv`: Test data as CSV file
- `results/`
    - (Pre)-trained modelss.
    - Results as CSV file.
- `.gitignore`: Command file for Github to ignore files with specific extensions. This
is useful to keep the repository clean. Templates for many programming languages are 
available [here](https://github.com/github/gitignore).

## Usage
The easiest way to reproduce the results is to use a service like 
[Binder](https://mybinder.org/) and run the Jupyter Notebook (if available).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TUD-STKS/sepfir_onset_detection/HEAD)

To run the scripts or to start the Jupyter Notebook locally, at first, please ensure 
that you have a valid Python distribution installed on your system. Here, at least 
Python 3.10 is required.

You can then call `run_jupyter-lab.ps1` or `run_jupyter-lab.sh`. This will install a new 
[Python venv](https://docs.python.org/3/library/venv.html), which is our recommended way 
of getting started.

To manually reproduce the results, you should create a new Python venv as well.
Therefore, you can run the script `create_venv.sh` on a UNIX bash or `create_venv.ps1`
that will automatically install all packages from PyPI. Afterwards, just type 
`source .virtualenv/bin/activate` in a UNIX bash or `.virtualenv/Scripts/activate.ps1`
in a PowerShell.

The individual steps to reproduce the results should be in the same order as in the 
paper. Great would be self-explanatory names for each step.

At first, we import required Python modules and load the dataset, which is already 
stored in `data`. 

```python
from file_handling import (load_data)


training_data = load_data("../data/train.csv")
```

Since the data is stored as a Pandas dataframe, we can theoretically multiple features. 
Here, we restrict the data features to the living area. With the function 
`select_features`, we obtain numpy arrays and the feature transformer that can also be
used for transforming the test data later. Next, we normalize them to zero mean and 
unitary variance.

```python
from sklearn.preprocessing import StandardScaler
from preprocessing import select_features


X, y, feature_trf = select_features(
    df=training_data, input_features=["GrLivArea"], target="SalePrice")
scaler = StandardScaler().fit(X)
X_train = scaler.transform(X)
y_train = y
```

We optimize a model using a random search.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform

from pyrcn.extreme_learning_machine import ELMRegressor


model = RandomizedSearchCV(
    estimator=ELMRegressor(input_activation="relu", random_state=42,
                           hidden_layer_size=50),
    param_distributions={"input_scaling": uniform(loc=0, scale=2),
                         "bias_scaling": uniform(loc=0, scale=2),
                         "alpha": loguniform(1e-5, 1e1)},
    random_state=42, n_iter=200, refit=True).fit(X, y)
```

We load and transform test data.

```python
from file_handling import load_data


test_data = load_data("../data/test.csv")
X = feature_trf.transform(test_data)
X_test = scaler.transform(X)
```

Finally, we predict the test data.

```python
y_pred = model.predict(X_test)
```

After you finished your experiments, please do not forget to deactivate the venv by 
typing `deactivate` in your command prompt.

The aforementioned steps are summarized in the script `main.py`. The easiest way to
reproduce the results is to either download and extract this Github repository in the
desired directory, open a Linux Shell and call `run.sh` or open a Windows PowerShell and
call `run.ps1`. 

In that way, again, a [Python venv](https://docs.python.org/3/library/venv.html) is 
created, where all required packages (specified by `requirements.txt`) are installed.
Afterwards, the script `main.py` is excecuted with all default arguments activated in
order to reproduce all results in the paper.

If you want to suppress any options, simply remove the particular option.

## Acknowledgements
This research was supported by
```
Nobody
```


## License and Referencing
This program is licensed under the BSD 3-Clause License. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

More information about licensing can be found [here](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)
and [here](https://en.wikipedia.org/wiki/License).

You can use the following BibTeX entry
```
@inproceedings{src:Steiner-22,
  author    = "Peter Steiner",
  title     = "Template Repository for Research Papers",
  booktitle = "Proceedings of the Research Seminar",
  year      = 2022,
  pages     = "1--6",
}
```
## Appendix
For any questions, do not hesitate to open an issue or to drop a line to [Peter Steiner](mailto:peter.steiner@tu-dresden.de)
