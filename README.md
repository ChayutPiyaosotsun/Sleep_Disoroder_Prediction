# Sleep Disorder Prediction

This project is focused on predicting sleep disorders based on various health and lifestyle attributes. It utilizes the [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/) from Kaggle.

## Dataset Overview

The dataset provides insights into individuals' sleeping habits and related lifestyle factors. The attributes include gender, age, occupation, sleep duration, quality of sleep, physical activity level, stress level, BMI category, heart rate, and daily steps. Using this data, the aim is to predict whether an individual has a sleep disorder.

## Getting Started

### Prerequisites

- Python 3.x
- Pandas
- Scikit-learn
- Joblib
- Imbalanced-learn

You can install the required packages using pip:

```
pip install pandas scikit-learn joblib imbalanced-learn
```

### Usage

1. **Data Preparation**:
    - Download the dataset from [here](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/).
    - Save the dataset in a `data/` directory within the project folder.

2. **Training**:
    - Run the training script to preprocess the data, train the model, and save it:
    ```
    python train.py
    ```

3. **Prediction**:
    - Use the provided prediction script to make predictions on new data points:
    ```
    python predict.py
    ```

## Model

The project uses the RandomForestClassifier from Scikit-learn for training the model. Data preprocessing steps include encoding categorical variables, standardizing numerical attributes, and handling class imbalances using SMOTE.

## Resources

- [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset/)

## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request.

## Acknowledgments

- Thanks to Kaggle and the dataset creators for providing the Sleep Health and Lifestyle dataset.
