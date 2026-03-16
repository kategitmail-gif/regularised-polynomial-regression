# Statistical Machine Learning – MSc Practical

## Regularised Polynomial Regression


---

## Overview

In this exercise we study **linear regression and polynomial regression** in the context of statistical machine learning.

The dataset consists of:

* a **training set**
  [
  (x_i, y_i)_{i=1,\dots,n}
  ]

* a **test set**
  [
  (\tilde{x}_i, \tilde{y}*i)*{i=1,\dots,n}
  ]

Each dataset is composed of pairs of inputs and outputs ((x_i, y_i)), where

* (x_i \in \mathbb{R})
* (y_i \in \mathbb{R})

These pairs are realizations of **independent and identically distributed random variables** ((X, Y)).

---

## Learning Objective

Let the **risk under the squared loss function** be

[
R(h_\beta) = \mathbb{E}[(Y - h_\beta(X))^2]
]

Our goal is to learn a prediction rule with **small generalisation risk**.

We consider an input expansion

[
\phi : \mathbb{R} \rightarrow \mathbb{R}^{M+1}
]

and a linear prediction rule of the form

[
h_\beta(x) = \phi(x)^T \beta
]

---

## Empirical Risk Minimisation

The empirical risk minimiser is

[
h_{\hat{\beta}}(x) = \phi(x)^T \hat{\beta}
]

where

[
\hat{\beta} = \arg \min_{\beta \in \mathbb{R}^2} J(\beta)
]

with objective function

[
J(\beta) =
\frac{1}{n} \sum_{i=1}^{n} (y_i - \phi(x_i)^T \beta)^2
======================================================

\frac{1}{n} |y - \Phi \beta|^2
]

---

## Matrix Notation

The response vector is

[
y = (y_1, \dots, y_n)^T
]

and the design matrix is

[
\Phi =
\begin{pmatrix}
\phi(x_1)^T \
\phi(x_2)^T \
\vdots \
\phi(x_n)^T
\end{pmatrix}
]

---

## Python Implementation

The following Python code imports the required libraries and loads the dataset.

The practical explores:

* **Simple Linear Regression**
* **Gradient Descent optimisation**
* **Stochastic Gradient Descent**
* **Polynomial Regression**
* **Ridge Regularisation**
* **Over-parameterisation and Double Descent**

---

## Author

Kate Leontieva

