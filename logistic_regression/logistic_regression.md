# Logistic Regression

In binary classification, we want to separate our data into two classes that we'll label 0 and 1. Examples of this include, among many others, classification of tumours as malign or benign or of emails as spam or not spam. To be concrete, for each input $x^{(i)}$, $i\in\{0,1,2,\ldots,m\}$, we wish to predict a corresponding label $y^{(i)}$, where $y^{(i)}\in\{0,1\}$. 

Given we know that $y\in\{0,1\}$ it makes sense to modify our hypothesis to reflect this. There are many functions that would do the job, but logistic regression uses the sigmoid function (also known as the logistic function), defined as 
$$
g(z) = \frac{1}{1-\mathrm{e}^{-z}}.
$$
The form of our hypothesis $h$, parameterised by $\theta$, is now 
$$
h_\theta(x) = g(\theta^Tx) = \frac{1}{1-\mathrm{e}^{-\theta^Tx}},
$$
which returns values in $[0,1]$ as required. 

We are looking for a set of parameters $\theta$ that predict the positive class (i.e. the ones labelled 1) and as such we can assume
$$
P(y=1\mid x;\theta) = h_\theta(x)\qquad\text{and}\qquad P(y=0\mid x;\theta) = 1 - h_\theta(x)
$$
since if an instance is labelled 1 it can't be labelled 0, and vice versa. This can be written more compactly as 
$$
p(y\mid x;\theta) = h_\theta(x)^y(1-h_\theta(x)^{1-y}),
$$
which is the distribution of $y$ given $x$, parameterised by $\theta$. Assuming the training examples were generated independently, the likelihood is given by
$$
L(\theta) = p(\vec{y}\mid X;\theta) = \prod_{i=1}^{m}h(x)^{y}(1-h(x))^{1-y},
$$
where $i$ indexes each of the $x$ and $y$ in the product. Given inputs $X$ we'd like to maximise the likelihood. In general a sum is easier to deal with than a product so in practice we actually take logs of $L$ to form a sum of logarithms and maximise this instead (this works since the logarithm is a monotonically increasing function, so maximising $\log f(x)$ is equivalent to maximising $f(x)$).

Taking logs so that $\ell (\theta) = \log L(\theta)$ and differentiating $\ell$ with respect to a single parameter $\theta_j$ for a single training example, we can show that
$$
\frac{\partial \ell}{\partial\theta_j} = (y - g(\theta^Tx))x_j = (y - h(x))x_j,
$$
leading to the following update rule for $\theta$:
$$
\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}.
$$

## Testing
To test the algorithm, we can use a toy dataset built from the scikit-learn library that consists of 500 data points representing two features and two classes. This is split into training and test sets and fed into the algorithm to obtain a decision boundary. This recovered 