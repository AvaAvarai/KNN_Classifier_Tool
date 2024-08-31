# KNN_Classifier_Tool

Tool to test KNN classification algorithm with different K values, distance metrics, preprocessing steps, and training splits.

## Features

- Load data from CSV file
- Choose between different preprocessing steps:
  - None
  - Standardization
  - Normalization
- Choose between different distance metrics:
  - Euclidean
  - Manhattan
  - Minkowski (with customizable p value)
- Choose between different K values
- View confusion matrix
- View classification report (accuracy, precision, recall, F1 score)

## Data Format

The data should be in a CSV file with the target variable in a column named 'class', case-insensitive. The rest of the columns are numerical features.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
