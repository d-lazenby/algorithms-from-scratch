# Common Machine Learning Algorithms From Scratch
This repository contains some common machine learning algorithms implemented from scratch in Python. This is done as a learning exercise and the complexity will increase as more algorithms are added. I'll add some brief notes on the mathematics (e.g. main gist of the algorithm and derivations of gradient updates) as and when I have time to write them up, again as a learning exercise for myself. I've tried to be consistent with notation and largely follow that used in Andrew Ng's Stanford ML course, which I'm working through while I develop the repo.

Algorithms covered so far:
- Linear Regression,
- Logistic Regression.

### Notation and terminology used in the notes
For a given supervised problem we'll take some training data $(x,y)$, where $x$ are the inputs or features and $y$ are the outputs, labels or targets, and a hypothesis $h$ (essentially the assumed form of the relationship between $x$ and $y$) dependent on a set of parameters $\theta$ and modify the latter such that $h_\theta\approx y$ for the training data, where the dependency of $h$ upon the parameters is indicated by the subscript. We use $m$ and $n$ to respectively denote the number of training examples and number of features, and refer to a single training example as $(x^{(i)}, y^{(i)})$, where $(x,y) = \{(x^{(i)}, y^{(i)});\ i\in\{1,\ldots,m\}\}$. 

The process of training involves iteratively modifying $\theta$ until a "good enough" approximation $h_\theta\approx y$ is found. We do this by defining a cost function $J(\theta)$ for the problem, which quantifies the error in our predicted outputs from the real outputs and the task of the algorithm is thus to
$$
\operatorname*{minimize}_\theta\ J(\theta).
$$