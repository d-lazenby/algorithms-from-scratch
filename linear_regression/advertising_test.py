# Data from https://www.kaggle.com/code/devzohaib/simple-linear-regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn import datasets

from linear_regression import LinearRegression, mse

def standardize(arr, mean, std):
    return (arr - mean) / std

def main():
    tv = pd.read_csv('./linear_regression/tvmarketing.csv')
    # Have checked – 0 nulls, shape (200,2)
    # Simple scatter shows no major outliers
    tv = tv.astype({'TV': 'float32', 'Sales': 'float32'})

    X = tv[['TV']].copy()
    y = tv['Sales'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

    # Standardize
    mean, std = np.mean(X_train), np.std(X_test)
    X_train_std = standardize(X_train, mean, std)
    X_test_std = standardize(X_test, mean, std)

    # Train model
    model = LinearRegression(lr=0.01, n_iter=1000)
    model.fit(X_train_std, y_train)
    print(model.theta0, model.theta1)
    
    # Performance
    predictions = model.predict(X_test_std)
    score = mse(predictions, y_test)

    # # Calculate two points on the predicted line
    Xmin, Xmax = standardize(np.min(X), mean, std), standardize(np.max(X), mean, std)
    y_pred_min, y_pred_max = model.predict(Xmin), model.predict(Xmax)
    
    # Plot linear model
    plt.plot([np.min(X), np.max(X)], [y_pred_min, y_pred_max], '-', label=r'$\theta_0$ + $\theta_1$X')

    # Plot training and test data
    plt.scatter(X_train, y_train, s=10, color='black', label='train')
    plt.scatter(X_test, y_test, s=10, color='red', label='test')

    plt.xlabel('TV')
    plt.ylabel('Sales')

    plt.title(f'Linear regression with #iterations = {model.n_iter} and learning rate = {model.lr}\nMSE = {round(score)}')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
