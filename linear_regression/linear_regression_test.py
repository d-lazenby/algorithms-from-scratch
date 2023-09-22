import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from linear_regression import LinearRegression, mse


def main():
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    fig, ax = plt.subplots(nrows=2, figsize=(10, 20))
    cmap = plt.get_cmap('viridis')
    X_max = np.max(X)

    n_iters = [100, 1000]
    for i, n_iter in enumerate(n_iters):
        ax[i].scatter(X_train, y_train, s=10, color=cmap(0.2))
        ax[i].scatter(X_test, y_test, s=10, color=cmap(0.7))

        lrs = [0.001, 0.01, 0.1]
        mses = []
        plot_lines = []
        for lr in lrs:
            regressor = LinearRegression(lr=lr, n_iter=n_iter)
            regressor.fit(X_train, y_train)
            predictions = regressor.predict(X_test)
            plot_line = regressor.predict(X)
            plot_lines.append(plot_line)
            mses.append(round(mse(y_test, predictions), 2))

        for pl, lr, m in zip(plot_lines, lrs, mses):
            ax[i].plot(X, pl, color='gray', lw=0.8, label=f"LR: {lr};\nMSE {m}")
            # Annotate the line with the label text
            pl_max = np.max(pl)
            ax[i].text(X_max, pl_max, f"LR: {lr}; MSE: {m}",
                       fontsize=10, color='black', va='center', ha='left')
        ax[i].set_title(f'Linear regression for various learning rates with #iterations = {n_iter}')
    plt.tight_layout()
    plt.savefig('results.png')
    # plt.show()


if __name__ == "__main__":
    main()

