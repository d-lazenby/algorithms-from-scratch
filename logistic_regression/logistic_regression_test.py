import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import make_blobs, make_classification
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression, accuracy

# Compute decision boundary
def compute_decision_boundary(X, model):
    return (model.threshold - model.theta0 - model.theta_j[1] * X ) / model.theta_j[0]

X, y = make_blobs(n_samples=500, n_features=2, centers=2, cluster_std=0.9, random_state=0)

# dataset = datasets.load_breast_cancer()
# X, y = dataset.data, dataset.target

# print(np.unique(y, return_counts=True))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)
df_train = pd.DataFrame(dict(X1=X_train[:,0], X2=X_train[:,1], y=y_train))
df_test = pd.DataFrame(dict(X1=X_test[:,0], X2=X_test[:,1], y=y_test))

model = LogisticRegression(lr=0.01)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy(y_test, predictions)}")

db_train = compute_decision_boundary(X_train[:,1], model)
db_test = compute_decision_boundary(X_test[:,1], model)

colors = {0:'red', 1:'blue'}
fig, ax = plt.subplots(ncols=2)
blobs_train = df_train.groupby('y')
for key, group in blobs_train:
    group.plot(ax=ax[0], kind='scatter', x='X1', y='X2', label=key, color=colors[key], alpha=0.5)

blobs_test = df_test.groupby('y')
for key, group in blobs_test:
    group.plot(ax=ax[1], kind='scatter', x='X1', y='X2', label=key, color=colors[key], alpha=0.5)

# Plot linear model
ax[0].plot(db_train, X_train[:,1], '-')
ax[1].plot(db_test, X_test[:,1], '-')
plt.show()
