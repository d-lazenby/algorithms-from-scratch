import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import make_blobs, make_classification
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression, accuracy

def compute_decision_boundary(X2, model):
    return (model.threshold - model.theta0 - model.theta_j[1] * X2 ) / model.theta_j[0]

def plot_results(Xtrain, ytrain, Xtest, ytest, model, accuracy=None, save=None):
    colors = {0:'red', 1:'blue'}
    fig, ax = plt.subplots(figsize=(10,4), ncols=2)

    for i, ml_set in enumerate(((Xtrain, ytrain), (Xtest, ytest))):
        print("Index is ", i)
        df = pd.DataFrame(dict(X1=ml_set[0][:,0], X2=ml_set[0][:,1], y=ml_set[1]))
        db = compute_decision_boundary(ml_set[0][:,1], model)
        print(f"Db computed fine for index {i}")
        blobs = df.groupby('y')

        for key, group in blobs:
            group.plot(ax=ax[i], kind='scatter', x='X1', y='X2', label=key, color=colors[key], alpha=0.5)

    # Plot linear model
        ax[i].plot(db, ml_set[0][:,1], '-', label=r'$h_\theta$')
        ax[i].legend()

    ax[0].set_title("Train")
    if accuracy:
        ax[1].set_title(f"Test. Accuracy {accuracy}")
    else:
        ax[1].set_title(f"Test")

    if save:
        plt.savefig(f'./logistic_regression/{save}.png')
    
    plt.show()


# random_state = 8
# X, y = make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=2.0, random_state=12)
X, y = make_classification(n_samples=500, n_features=2, n_informative=2, 
                           n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1, 
                           class_sep=0.5, random_state=5)
# sklearn.datasets.make_classification(n_samples=100, n_features=20, *, 
#                                      n_informative=2, n_redundant=2, n_repeated=0, 
#                                      n_classes=2, n_clusters_per_class=2, weights=None, 
#                                      flip_y=0.01, class_sep=1.0, 
#                                      hypercube=True, shift=0.0, scale=1.0, shuffle=True, 
#                                      random_state=None)[source]

# dataset = datasets.load_breast_cancer()
# X, y = dataset.data, dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

model = LogisticRegression(lr=0.01)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
acc = accuracy(y_test, predictions)

print(f"Accuracy: {acc}")

plot_results(X_train, y_train, X_test, y_test, model, accuracy=acc)
