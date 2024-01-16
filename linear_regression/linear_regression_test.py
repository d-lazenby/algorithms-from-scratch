import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from linear_regression import LinearRegression, mse

def main():
    # Get data 
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
    
    # Train model
    model = LinearRegression(lr=0.01, n_iter=10)
    model.fit(X_train, y_train)
    
    # Performance
    predictions = model.predict(X_test)
    score = mse(predictions, y_test)

    # Calculate two points on the predicted line
    Xmin, Xmax = np.min(X), np.max(X)
    y_pred_min, y_pred_max = model.predict(Xmin), model.predict(Xmax)
    
    # Plot linear model
    plt.plot([Xmin, Xmax], [y_pred_min, y_pred_max], '-', label=r'$\theta_0$ + $\theta_1$X')

    # Plot training and test data
    plt.scatter(X_train, y_train, s=10, color='black', label='train')
    plt.scatter(X_test, y_test, s=10, color='red', label='test')
    
    plt.xlabel('$X$')
    plt.ylabel('$y$')

    plt.title(f'Linear regression with #iterations = 1000 and learning rate = 0.01\nMSE = {round(score)}')
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()

