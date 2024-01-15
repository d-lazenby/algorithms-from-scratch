# Logistic Regression

In binary classification, we want to separate our data into two classes that we'll label 0 and 1. Examples of this include, among a great many others, a spam email classifier and classification of tumours as malign or benign. To be concrete, given an input $x^{(i)}$, $i\in\{0,1,2,\ldots,m\}$, we wish to predict a corresponding label $y^{(i)}$, where $y^{(i)}\in\{0,1\}\ \forall i$. 

Given we know that $y\in\{0,1\}$ it makes sense to modify our hypothesis to reflect this. There are many functions that would do the job, but logistic regression uses the sigmoid function (also known as the logistic function), defined as 
$$
g(z) = \frac{1}{1-\mathrm{e}^{-z}}.
$$
The form of our hypothesis $h$, parameterised by a $\theta$, is now 
$$
h_\theta(g(\theta^Tx)) = \frac{1}{1-\mathrm{e}^{-\theta^Tx}},
$$
which returns values in $[0,1]$ as required. 

We are looking for a set of parameters $\theta$ that predict the positive class (i.e. the ones labelled 1) and as such we can assume
$$
P(y=1\mid x;\theta) = h_\theta(x)\qquad\text{and}\qquad P(y=0\mid x;\theta) = 1 - h_\theta(x)
$$
since if an instance is not labelled 1 it must be labelled 0, and vice versa. This can be written more compactly as 
$$
P(y\mid x;\theta) = h_\theta(x)^y(1-h_\theta(x)^{1-y}).
$$
