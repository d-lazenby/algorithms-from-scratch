# Logistic Regression

In binary classification, we want to separate our data into two classes that we'll label 0 and 1. Examples of this include, among many others, classification of tumours as malign or benign or of emails as spam or not spam. To be concrete, given an input $x^{(i)}$, $i\in\{0,1,2,\ldots,m\}$, we wish to predict a corresponding label $y^{(i)}$, where $y^{(i)}\in\{0,1\}$. 

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
which is the distribution of y given x, parameterised by $\theta$. Assuming the training examples were generated independently, the likelihood is given by
$$
L(\theta) = p(\vec{y}\mid X;\theta) = \prod_{i=1}^{m}h(x)^{y}(1-h(x))^{1-y},
$$
where $i$ indexes each of the $x$ and $y$ in the product.

Taking logs so that $\ell (\theta) = \log L(\theta)$ and differentiating $\ell$ with respect to a single parameter $\theta_j$ for a single training example, we can show that
$$
\frac{\partial \ell}{\partial\theta_j} = (y - g(\theta^Tx))x_j = (y - h(x))x_j,
$$
leading to the stochastic gradient ascent (*ascent* because we want to maximise the log-likelihood) rule
$$
\theta_j := \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)}))x_j^{(i)}.
$$

## Testing
Cancer dataset data split: (array([0, 1]), array([212, 357]))