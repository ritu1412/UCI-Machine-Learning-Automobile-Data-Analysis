# Automobile Data Analysis

This repository contains code for analyzing automobile data, focusing on continuous features and their relationship with the price of the automobiles. The analysis includes data cleaning, feature selection, visualization, and regression modeling.

## Data Source

The data used in this project is fetched from the UCI Machine Learning Repository, specifically the "Automobile" dataset.

## Dependencies

To run the code in this repository, install dependencies from requirements.txt file

## Code Overview

The analysis is performed in the `automobile.ipynb` Jupyter notebook, which includes the following steps:

1. **Importing Libraries**: Essential libraries for data analysis and machine learning are imported.

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import sklearn
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from scipy.stats import pearsonr
    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import cross_val_predict, KFold
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import RepeatedKFold
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    ```

2. **Fetching Data**: The automobile dataset is fetched from the UCI Machine Learning Repository.

    ```python
    automobile = fetch_ucirepo(id=10)
    df_features = pd.DataFrame(automobile.data.features)
    df_targets = pd.DataFrame(automobile.data.targets)
    ```

3. **Data Cleaning**: Non-continuous features and missing values are removed. The remaining data is converted to the `float64` type for analysis.

    ```python
    continuous_features = ["wheel-base", "length", "width", "height", "curb-weight", "engine-size", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
    df_features = df_features[continuous_features]
    df_features = df_features.dropna(subset=["price"])
    df_features = df_features.dropna()
    df_features = df_features.astype(np.float64)
    ```

4. **Exploratory Data Analysis (EDA)**: Scatter plots are generated for each feature against the target variable (price).

    ```python
    def plot_scatter(features, target):
        fig, axes = plt.subplots(len(features.columns)//3, 3, figsize=(15, 20))
        for i, feature in enumerate(features.columns):
            row, col = divmod(i, 3)
            axes[row, col].scatter(features[feature], target)
            axes[row, col].set_xlabel(feature)
            axes[row, col].set_ylabel("Price")
        plt.tight_layout()
        plt.show()

    plot_scatter(features, target)
    ```

5. **Regression Modeling**: Linear regression and ridge regression models are applied to predict the price of the automobiles. Cross-validation is used to evaluate model performance.

    ```python
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("Linear Regression R^2 Score:", r2_score(y_test, y_pred))
    ```

6. **Cross-Validation**: Cross-validation with K-Fold and repeated K-Fold strategies are employed to validate the model.

    ```python
    cv = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(lr, features, target, cv=cv, scoring='r2')
    print("Cross-Validation R^2 Scores:", cv_results)
    print("Mean R^2 Score:", np.mean(cv_results))
    ```

## Usage

To run the analysis, open the `automobile.ipynb` Jupyter notebook and execute the cells sequentially. Ensure that the required libraries are installed and the data is correctly fetched from the UCI repository.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.