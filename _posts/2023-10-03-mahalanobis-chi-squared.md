---
layout: post
title: The Relationship between the Mahalanobis Distance and the Chi-Squared Distribution
date: 2023-10-03 09:56:00-0400
modified:
categories: [stats, ml]
description: The Relationship between the Mahalanobis Distance and the Chi-Squared Distribution
tags: mahalanobis chi2
image:
  feature: 1141s.jpg
  credit: Designed by onlyyouqj / Freepik
  creditlink: https://www.freepik.com
comments: true
share:
permalink: /mahalanbis-chi-squared/
giscus_comments: true
related_posts: false
related_publications: ThillKonenBaeck2018_1000097489, Thill2017c
---

In practice, sometimes (multivariate) Gaussian distributions are used for anomaly detection tasks (assuming that the considered data is approx. normally distributed): the parameters of the Gaussian can be estimated using maximum likelihood estimation (MLE) where the maximum likelihood estimate is the sample mean and sample covariance matrix. After the estimating the parameters of the distribution one has to specify a critical value which separates the normal data from the anomalous data. Typically, this critical value is taken from the probability density function (PDF) in such a way that it is smaller than the PDF value of all normal data points in the data set. Then a new data point can be classified as anomalous if the value of the PDF for this new point is below the critical value. Hence, the critical value specifies a boundary which is used to separate normal from anomalous data. In the univariate case the boundary separates the lower and upper tails of the Gaussian from its center (mean). In the 2-dimensional case the boundary is an ellipse around the center and in higher dimensions the boundary can be described by an ellipsoid.
But what do we do, if want to find a boundary in a way that separates the most unlikely 2% of the data points from a sample from the remaining 99%. In the univariate case, this scenario is simple: We just have to compute the first percentile (1% quantile) and 99th percentile. All points that end up in the specified tails would then be classified as anomalous. For the multivariate case this is not that straightforward any longer, since our boundary has to be described by an ellipsoid. However, there is a way out of this problem, which has to do with a so called Mahalanobis-distance, as we will see in the following.

<!--more-->
\\(
   \def\matr#1{\mathbf #1}
   \def\tp{\mathsf T}
\\)

## Prerequisites

### Multiplication of a Matrix with its Transpose
Generally, the product of a $$n \times \ell$$ matrix $$\matr A$$ and a $$\ell \times p$$ matrix $$\matr B$$ is defined as:

$$
(\matr A \matr B)_{ij}=\sum_{k=1}^m \matr A_{ik} \matr B_{kj}
$$

Then, the multiplication of a matrix $$\matr A$$ with its transpose $$\matr A^\tp$$ can be written as:

$$
\begin{align}
(\matr A \matr A^\mathsf T)_{ij} &= \sum_{k=1}^\ell \matr A_{ik} \matr A^\mathsf T_{kj} \\
&= \sum_{k=1}^\ell \matr A_{ik} \matr A_{jk} \\
\matr A \matr A^\mathsf T &= \sum_{k=1}^\ell \vec a_{k} \vec a_{k}^\tp \label{eq:matrixProductWithTranspose}
\end{align}
$$

where $$\vec a_{k}$$ is the $$k$$th column vector of matrix $$\matr A$$.
Another trivial relation required later is as follows:

$$
\begin{align}
x &= \vec a^\mathsf T \vec b \\
y &= \vec b^\mathsf T \vec a = x^\mathsf T=x \\
xy &= \vec a^\mathsf T \vec b \, \vec b^\mathsf T \vec a = x^2 \\
&= (\vec a^\mathsf T \vec b)^2 \label{eq:multOfTwoSkalars}
\end{align}
$$

### Inverse of a Matrix-Product
$$
\begin{align}
(\matr{A} \matr B)^{-1} = \matr B^{-1}\matr A^{-1} \label{eq:inverseProduct}
\end{align}
$$
since
$$
\begin{align}	( \matr A  \matr B)(\matr B^{-1} \matr  A^{-1})
&= (\matr A(\matr  B  \matr B^{-1})) \matr A^{-1} \\
&= ( \matr  A\mathbf{I}) \matr A^{-1} \\&= \matr A  \matr A^{-1} \\&= \mathbf{I}
\end{align}
$$

### Eigenvalues and Eigenvectors
For a matrix $$\matr A $$ solve:
$$
\begin{equation}
  \matr{A} \vec{u} = \lambda \vec{u}
\end{equation}
$$

A value $$\lambda $$ which fulfills the equation is called an eigenvalue of $$\matr A$$ and the corresponding vector $$\vec\mu$$  is called eigenvector. When the eigenvectors of $$\matr A$$ are arranged in a matrix $$\matr U$$, we have:

$$
\begin{align}
\matr{A} \matr{U} &=
\matr{U} \matr{\Lambda}\\
\matr{A} &= \matr{U} \matr{\Lambda} \matr{U}^{-1} \label{eq:eigendecomp}
\end{align}
$$

where $$\matr \Lambda$$ is a diagonal matrix containing the eigenvalues $$\lambda_i$$ of the corresponding eigenvectors. This representation, where the matrix is represented in terms of its eigenvalues and eigenvectors is also called eigenvalue decomposition. For symmetric matrices $$\matr{A}$$, the eigenvectors are orthogonal (orthonormal) and the matrix $$\matr{U}$$ is orthogonal as well (the product with its transpose is the identity matrix). In this case $$\matr{U}^{-1}=\matr{U}^{T}$$, and equation $$\eqref{eq:eigendecomp}$$ can be written as:

$$
\begin{align}
\matr{A} &= \matr{U} \matr{\Lambda} \matr{U}^{T}
\end{align}
$$

In this case, also the square root of $$\matr A$$ (written here as $$\matr A^{\frac{1}{2}}$$) – such that $$\matr A^{\frac{1}{2}}\matr A^{\frac{1}{2}}=A$$ – can be easily found to be:

$$
\begin{align}
\matr A^{\frac{1}{2}} &= \matr{U} \matr{\Lambda}^{\frac{1}{2}} \matr{U}^{T} \label{eq:sqrtSymMatrix}
\end{align}
$$

since

$$
\begin{align}
\matr A^{\frac{1}{2}} \cdot \matr A^{\frac{1}{2}} &= \matr{U} \matr{\Lambda}^{\frac{1}{2}} \matr{U}^{T} \matr{U} \matr{\Lambda}^{\frac{1}{2}} \matr{U}^{T} \\
&=\matr{U} \matr{\Lambda}^{\frac{1}{2}} \matr I \matr{\Lambda}^{\frac{1}{2}} \matr{U}^{T} \\
&= \matr{U} \matr{\Lambda} \matr{U}^{T} \\
&= \matr A
\end{align}
$$

The eigenvalue decomposition of the inverse of a matrix $$\matr A$$ can be computed as follows, using the relation described in equation $$\eqref{eq:inverseProduct}$$ and the associative property of the matrix product:

$$
\begin{align}
\matr{A}^{-1} &= \big( \matr U \matr \Lambda \matr U^{-1} \big) \\
&= \big(  \matr U^{-1} \big)^{-1} \matr \Lambda^{-1}   \matr U^{-1}\\
&=  \matr U \matr \Lambda^{-1}   \matr U^{-1}\\
&=  \matr U \matr \Lambda^{-1}   \matr U^{T}\label{eq:eigenvalueInverse} \\
\end{align}
$$

Note that $$\Lambda^{-1} $$ is again a diagonal matrix containing the inverse eigenvalues of $$ \matr{A}$$.

### Linear Affine Transform of a Normally Distributed Random Variable

Assume we apply a linear affine transform to a random variable $$X \thicksim N(\vec \mu_x, \Sigma_x)$$ with a mean vector $$\vec\mu_x$$  and a covariance matrix $$\Sigma_x$$ in order to create a new random variable $$Y$$:

$$
Y=\matr A X + \vec b.
$$

One can compute the new mean $$\vec\mu_y$$ and covariance matrix $$\Sigma_y$$ for $$Y$$:

$$
\begin{align}
\vec \mu_y &= E \{ Y \} \\
&=E \{\matr A X + \vec b  \} \\
&= \matr A E \{\matr  X \} + \vec b \\
&= \matr A \vec \mu_x + \vec b \label{eq:AffineLinearTransformMean} \\
\end{align}
$$

and

$$
\begin{align}
\matr \Sigma_y &= E \{ (Y - \vec \mu_y) (Y - \vec \mu_y)^\tp \} \\
&= E \{ \big[(\matr A X + \vec b) - (\matr A \vec \mu_x + \vec b) \big] \big[(\matr A X + \vec b) - (\matr A \vec \mu_x + \vec b) \big]^\tp \} \\
&= E \{ \big[\matr A (X -  \vec \mu_x) \big] \big[\matr A (X -  \vec \mu_x  ) \big]^\tp \} \\
&= E \{ \matr A (X -  \vec \mu_x) (X -  \vec \mu_x  )^\tp \matr A^\tp \} \\
&= \matr A E \{  (X -  \vec \mu_x) (X -  \vec \mu_x  )^\tp  \}  \matr A^\tp\\
&= \matr A \matr \Sigma_x \matr A^\tp \label{eq:AffineLinearTransformCovariance}\\
\end{align}
$$

## Quantile Estimation for multivariate Gaussian Distributions
- Calculating quantiles for multivariate normal distributions is not that trivial as in the one-dimensional case, since we cannot simply compute the integral in the tails of the distribution
- The quantiles in the bivariate case can be seen as ellipses, in higher dimensions as ellipsoids
- The Mahalanobis distance is an interesting measure to describe all points on the surface of an ellipsoid.
- More formal: The usual quantile definition requires a random variable: The p-quantile for a random distribution is the value that fulfills . In the case of a multivariate normal distribution we can take the squared Mahalanobis distance  between a point of the multivariate normal distribution and its mean as such a random variable. Then the p-quantile computation will answer the following question: Which value  is required so that a random point  fulfills ? In other words, when we pick a random point  from the distribution, it will have with probability p a squared Mahalanobis distance equal or smaller than . The set of points with  forms an ellipsoid.  
- In a naive solution one can use a Monte Carlo approach to sample the multivariate normal distribution and compute the quantile based on the Mahalanobis distances of the elements of the sample
- However, this Monte Carlo approach is rather computationally inefficient, especially if quantiles have to be computed very often
- One can show that the squared Mahalanobis distance of a Gaussian distribution is actually Chi-Square distributed.

### Empirical Results suggesting that the Mahalanobis Distance is Chi-Square distributed
In a Quantile-Quantile Plot one can see that quantiles of the Mahalanobis distance of a sample drawn from a Gaussian distribution is very similar to the corresponding quantiles computed on the Chi-Square distribution. The following R-script shows this:

{% highlight R %}
library(Matrix)
library(MASS)
library(ggplot2)
DIM = 10
nSample = 1000

Posdef <- function (n, ev = runif(n, 0, 1))
{
  Z <- matrix(ncol=n, rnorm(n^2))
  decomp <- qr(Z)
  Q <- qr.Q(decomp)
  R <- qr.R(decomp)
  d <- diag(R)
  ph <- d / abs(d)
  O <- Q %*% diag(ph)
  Z <- t(O) %*% diag(ev) %*% O
  return(Z)
}

Sigma = Posdef(DIM)
muhat = rnorm(DIM)


sample <- mvrnorm(n=nSample, mu = muhat, Sigma = Sigma)
C <- .5*log(det(2*pi*Sigma))
mahaDist2 <- mahalanobis(x=sample, center=muhat,cov=Sigma)

#
# Interestingly, the Mahalanobis distance of samples follows a Chi-Square distribution
# with d degrees of freedom
#
pps <- (1:100)/(100+1)
qq1 <- sapply(X = pps, FUN = function(x) {quantile(mahaDist2, probs = x) })
qq2 <-  sapply(X = pps, FUN = qchisq, df=ncol(Sigma))

dat <- data.frame(qEmp= qq1, qChiSq=qq2)
p <- ggplot(data = dat) + geom_point(aes(x=qEmp, y=qChiSq)) +
  xlab("Sample quantile") +
  ylab("Chi-Squared Quantile") +
  geom_abline(slope=1)
plot(p)

{% endhighlight %}
<!--- %* -->

![Picture description]({{ site.url }}/images/Q-Q-Plot.png){: .image-center }


### The squared Mahalanobis Distance follows a Chi-Square Distribution: More formal Derivation
The Mahalanobis distance between two points $$\vec x$$ and $$\vec y$$ is defined as

$$
d(\vec x,\vec y) = \sqrt{(\vec x -\vec y )^\tp \matr \Sigma^{-1} (\vec x - \vec y)}
$$

Thus, the squared Mahalanobis distance of a random vector $$\matr X$$ and the center $$\vec \mu$$ of a multivariate Gaussian distribution is defined as:
$$
\begin{align}
D = d(\matr X,\vec \mu)^2 = (\matr X -\vec \mu )^\tp \matr \Sigma^{-1} (\matr X - \vec \mu ) \label{eq:sqMahalanobis}
\end{align}
$$

where $$\matr \Sigma$$ is a $$\ell \times \ell$$ covariance matrix and $$\vec \mu$$ is the mean vector. In order to achieve a different representation of $$D$$ one can first perform an eigenvalue decomposition on $$\matr \Sigma^{-1}$$ which is (with Eq. $$\eqref{eq:eigenvalueInverse}$$ and assuming orthonormal eigenvectors):

$$
\begin{align}
\matr \Sigma^{-1} &= \matr U \matr \Lambda^{-1} \matr U^{-1} \\
&= \matr U \matr \Lambda^{-1} \matr U^{T} \\
\end{align}
$$

With Eq. $$\eqref{eq:matrixProductWithTranspose}$$ we get:

$$
\begin{align}
\matr \Sigma^{-1} &= \sum_{k=1}^\ell \lambda_k^{-1} \vec u_{k} \vec u_{k}^\tp \label{eq:SigmaInverseAsSum}
\end{align}
$$

where $$\vec u_{k}$$ is the $$k$$th eigenvector of the corresponding eigenvalue $$\lambda_k$$. Plugging \eqref{eq:SigmaInverseAsSum} back into \eqref{eq:sqMahalanobis} results in:

$$
\begin{align}
D &= (\matr X -\vec \mu )^\tp \matr \Sigma^{-1} (\matr X - \vec \mu ) \\
 &= (\matr X -\vec \mu )^\tp \Bigg( \sum_{k=1}^\ell \lambda_k^{-1} \vec u_{k} \vec u_{k}^\tp \Bigg) (\matr X - \vec \mu ) \\
&= \sum_{k=1}^\ell \lambda_k^{-1} (\matr X -\vec \mu )^\tp   \vec u_{k} \vec u_{k}^\tp  (\matr X - \vec \mu )
\end{align}
$$

With Eq. \eqref{eq:multOfTwoSkalars} one gets:

$$
\begin{align}
D &= \sum_{k=1}^\ell \lambda_k^{-1} \Big[ \vec u_{k}^\tp  (\matr X - \vec \mu ) \Big]^2\\
&= \sum_{k=1}^\ell  \Big[ \lambda_k^{-\frac{1}{2}} \vec u_{k}^\tp  (\matr X - \vec \mu ) \Big]^2\\
&= \sum_{k=1}^\ell Y_k^2
\end{align}
$$

where $$Y_k$$ is a new random variable based on an affine linear transform of the random vector $$\matr X$$. According to Eq. \eqref{eq:AffineLinearTransformMean} , we have $$\matr Z = (\matr X - \vec \mu ) \thicksim N(\vec 0,\Sigma)$$.  If we set $$ \vec a_{k}^\tp = \lambda_k^{-\frac{1}{2}} \vec u_{k}^\tp$$ then we get $$Y_k = \vec a_{k}^\tp \matr Z = \lambda_k^{-\frac{1}{2}} \vec u_{k}^\tp \matr Z$$. Note that $$Y_k$$ is now a random Variable drawn from a univariate normal distribution $$Y_k \thicksim N(0,\sigma_k^2)$$, where, according to \eqref{eq:AffineLinearTransformCovariance}:

$$
\begin{align}
\sigma_k^2 &= \vec a_{k}^\tp \Sigma \vec a_{k}\\
&= \lambda_k^{-\frac{1}{2}} \vec u_{k}^\tp \Sigma \lambda_k^{-\frac{1}{2}} \vec u_{k} \\
&= \lambda_k^{-1} \vec u_{k}^\tp \Sigma \vec u_{k} \label{eq:smallSigma}
\end{align}
$$

If we insert

$$
\begin{align}
\matr \Sigma &= \sum_{j=1}^\ell \lambda_j \vec u_{j} \vec u_{j}^\tp
\end{align}
$$

into Eq. \eqref{eq:smallSigma}, we get:
$$
\begin{align}
\sigma_k^2 &= \lambda_k^{-1} \vec u_{k}^\tp \Sigma \vec u_{k} \\
&= \lambda_k^{-1} \vec u_{k}^\tp \Bigg( \sum_{j=1}^\ell \lambda_j \vec u_{j} \vec u_{j}^\tp \Bigg) \vec u_{k} \\
&=   \sum_{j=1}^\ell \lambda_k^{-1} \vec u_{k}^\tp \lambda_j \vec u_{j} \vec u_{j}^\tp \vec u_{k}   \\
&=   \sum_{j=1}^\ell \lambda_k^{-1} \lambda_j \vec u_{k}^\tp  \vec u_{j} \vec u_{j}^\tp \vec u_{k}   \\
\end{align}
$$

Since all eigenvectors $$\vec u_{i}$$ are pairwise orthonormal the dotted products $$\vec u_{k}^\tp  \vec u_{j}$$ and $$\vec u_{j}^\tp \vec u_{k}$$ will be zero for $$j \neq k$$. Only for the case $$j = k$$ we get:
$$
\begin{align}
\sigma_k^2 &= \lambda_k^{-1} \lambda_k \vec u_{k}^\tp  \vec u_{k} \vec u_{k}^\tp \vec u_{k}   \\
&= \lambda_k^{-1} \lambda_k   ||\vec u_{k}||^2 ||\vec u_{k}||^2   \\
&= \lambda_k^{-1} \lambda_k   ||\vec u_{k}||^2 ||\vec u_{k}||^2   \\
&= 1,
\end{align}
$$

since the the norm $$||\vec u_{k}||$$ of a orthonormal eigenvector is equal to 1.
The squared Mahalanobis  distance can be expressed as:

$$
\begin{align}
D &= \sum_{k=1}^\ell Y_k^2
\end{align}
$$

where
$$
Y_k \thicksim N(0,1).
$$

Now the Chi-square distribution with $$\ell$$  degrees of freedom is exactly defined as being the distribution of a variable which is the sum of the squares of $$\ell$$  random variables being standard normally distributed.
Hence, $$D$$ is Chi-square distributed with $$\ell$$ degrees of freedom.

### Derivation based on the Whitening Property of the Mahalanobis Distance
Since the inverse $$\matr \Sigma^{-1}$$  of the covariance matrix $$\matr \Sigma$$  is also a symmetric matrix, its squareroot can be found – based on Eq. \eqref{eq:sqrtSymMatrix} – to be a symmetric matrix . In this case we can write the squared Mahalanobis distance as
$$
\begin{align}
D  &= (\matr X -\vec \mu )^\tp \matr \Sigma^{-1} (\matr X - \vec \mu ) \\
   &= (\matr X -\vec \mu )^\tp \matr \Sigma^{-\frac{1}{2}} \matr \Sigma^{-\frac{1}{2}} (\matr X - \vec \mu )\\
   &= \Big( \matr \Sigma^{-\frac{1}{2}} (\matr X -\vec \mu ) \Big)^\tp  \Big(\matr \Sigma^{-\frac{1}{2}} (\matr X - \vec \mu ) \Big) \\
   &= \matr Y^\tp \matr Y \\
   &= ||\matr Y||^2 \\
   &= \sum_{k=1}^\ell Y_k^2
\end{align}
$$

The multiplication $$\matr Y = \matr W \matr Z$$, with $$\matr W=\matr \Sigma^{-\frac{1}{2}}$$ and $$\matr Z= \matr X -\vec \mu $$ is typically reffered to as a whitening transform, where in this case $$\matr W=\matr \Sigma^{-\frac{1}{2}}$$ is the so called Mahalanobis (or ZCA) whitening matrix. $$\matr Y$$  has zero mean, since $$(\matr X - \vec \mu ) \thicksim N(\vec 0,\Sigma)$$. Due to the (linear) whitening transform the new covariance matrix $$\matr \Sigma_y$$ is the identity matrix $$\matr I$$, as shown in the following (using the property in Eq. \eqref{eq:AffineLinearTransformCovariance}):

$$
\begin{align}
\matr \Sigma_y &= \matr W \matr \Sigma \matr W^\tp \\
&= \matr \Sigma^{-\frac{1}{2}} \matr \Sigma \Big( \matr \Sigma^{-\frac{1}{2}} \Big)^\tp \\
&= \matr \Sigma^{-\frac{1}{2}} \Big(\matr \Sigma^{\frac{1}{2}}\matr \Sigma^{\frac{1}{2}} \Big) \Big( \matr \Sigma^{-\frac{1}{2}} \Big)^\tp \\
&= \matr \Sigma^{-\frac{1}{2}} \Big(\matr \Sigma^{\frac{1}{2}}\matr \Sigma^{\frac{1}{2}} \Big) \matr \Sigma^{-\frac{1}{2}} \\
&= \Big(\matr \Sigma^{-\frac{1}{2}} \matr \Sigma^{\frac{1}{2}} \Big) \Big(\matr \Sigma^{\frac{1}{2}} \matr \Sigma^{-\frac{1}{2}}\Big) \\
&= \matr I
\end{align}
$$

Hence, all elements $$Y_k$$ in the random vector $$\matr Y$$ are random variables drawn from independent normal distributions $$Y_k \thicksim N(0,1)$$, which leads us to the same conclusion as before, that $$D$$ is Chi-square distributed with $$\ell$$ degrees of freedom.
