# Titanic Survival Prediction

This repository contains Python code for analyzing the Titanic dataset and predicting passenger survival using logistic regression.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project aims to predict passenger survival on the Titanic based on various features using logistic regression. The dataset (`train.csv`) contains information about passengers' attributes and whether they survived the disaster.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python (version 3.1)
- Pandas
- Matplotlib
- Seaborn
- NumPy
- Scikit-Learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/dineshghadge2002/Titanic_ClassificationRegration.git
   cd titanic-survival-prediction
   ```

2. Install the required dependencies:

   ```bash
   pip install pandas matplotlib seaborn numpy scikit-learn
   ```

## Usage

1. Place the `train.csv` file in the project directory.

2. Run the Jupyter Notebook or Python script `titanic_analysis.ipynb` to perform data preprocessing, exploratory data analysis, and model training.

3. Review the generated visualizations and model evaluation metrics.

## Code Overview

The code performs the following steps:

- Loading and exploring the dataset.
- Visualizing categorical variables against survival using Seaborn count plots.
- Cleaning the data by dropping unnecessary columns.
- Handling missing age values by calculating the mean for each passenger class.
- Creating dummy variables for categorical features.
- Splitting the dataset into training and testing sets.
- Building a logistic regression model using Scikit-Learn.
- Evaluating the model's performance using a confusion matrix.

## Results

The logistic regression model achieved an accuracy of 84% on the test set. The confusion matrix provides insights into the model's performance, showing true positives, true negatives, false positives, and false negatives.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to submit a pull request.
