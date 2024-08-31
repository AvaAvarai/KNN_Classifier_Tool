# KNN_Classifier_Tool

Tool to test KNN classification algorithm with different K values, distance metrics, preprocessing steps, and training splits.

## Features

- Load data from CSV file
- Choose between different preprocessing steps:
  - None
  - Standardization
  - Normalization
- Choose between different distance metrics:
  - Euclidean (L2 norm)
  - Manhattan (L1 norm)
  - Minkowski (with customizable p value, p=2 is euclidean, p=1 is manhattan)
- Choose between different K values
- View confusion matrix
- View classification report (accuracy, precision, recall, F1 score)

## Data Format

The data should be in a CSV file with the target variable in a column named 'class', case-insensitive. The rest of the columns are numerical features.

## Installation

We recommend using a virtual environment to install the required packages, developed and tested using Python 3.12. Some systems may require using `python3` instead of `python` to call the python interpreter and create the virtual environment.  

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

At the global level, you can install the packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
