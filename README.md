# README

Small portoflio of Python projects, mainly in:
- Optimization
- Machine Learning

These projects are related to my industrial PhD supervised by A. Iouditski (Laboratoire Jean Kuntzmann) in collaboration with STMicroelectronics (Crolles, France):
G. Caron, *Statistical learning techniques: application to inverse design and optimization of silicon germanium heterojunction bipolar transistors*, 2024

### Work in Progress

#### 1. Models
Several different Neural Network Models

#### 2. Sensitivity Analysis
Computation of the gradient of outputs w.r.t. inputs, following [1] and some other sensitivity measures inspired by [2].

[1] Y. Dimopoulos, Paul Bourret, and Sovan Lek. *Use of some sensitivity criteria for choosing networks with good generalization ability*, Neural Processing Letters 2 (Dec. 1995), pp. 1–4.

[2] Jaime Pizarroso, José Portela, and Antonio Muñoz. *NeuralSens: Sensitivity Analysis of Neural Networks*, Journal of Statistical Software 102.7 (2022), pp. 1–36.

#### 3. BFGS Failure Robustness Improvement
The line search  amounts to finding α which minimizes the helper function:

$$\phi(\alpha) = f(x_k + \alpha p_k), \alpha > 0$$

To ensure $$s_k^T y_k > 0$$ and $$p_k^T = - B_k^{-1} \nabla f(x_k)$$, the strong Wolfe conditions [3] are used:
- Armijo condition: $$\phi(\alpha_k) \leq \phi(0) + c_1 \alpha_k \phi'(0)$$
- curvature condition: $$\left| \phi'(\alpha_k) \right| \leq c_2 \left| \phi'(0) \right| $$

where $$c_1 = 10^{-4}$$ and $$c_2 = 0.9$$, following [4].

An iterative scheme is used to find such an acceptable step length $\alpha_k$. For SciPy's BFGS implementation, it is the one by Wright and Nocedal [4].
However, failure may happen in $\phi(\alpha_j)$ for any candidate step $\alpha_j$. We adopt a wide definition of failure in the case of black-box optimization on a third-party simulator (e.g. TCAD):
- simulator crashes
- improper inputs (e.g. unrealistic or punchthrough HBT profiles)

They can be detected by monitoring the simulation time, the input, or the result of an intermediary simulation. We also make the following assumption:

If $$\exists \alpha_j > 0, x_k + \alpha_j p_k \in F$$, then $$\forall \alpha > \alpha_j, x_k + \alpha p_k \in F$$
with F the set of failing profiles.

To avoid the optimization procedure to stop when encountering such a failure point, we propose to set:

$$\phi(\alpha_j) = \lambda \phi(0), \lambda > 1$$

In that case, the Armijo condition is not respected in $\alpha_j$. This new value of $\phi(\alpha_j)$ is taken into consideration when finding the next candidate step length $\alpha_k$, following one of three schemes (cubic interpolation, quadratic interpolation or, most of the time, bisection).

<img width="400" alt="bisection_2" src="https://github.com/user-attachments/assets/0ed74f86-b5c8-4bcb-a6a9-bcf9cce2c886"> &emsp; &emsp;
<img width="370" alt="convergence" src="https://github.com/user-attachments/assets/0adb91cc-ddef-4b41-b932-f34d4c42f854">

Benchmark optimization functions from https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective?tab=readme-ov-file

[3] Philip Wolfe. “Convergence Conditions for Ascent Methods”. In: SIAM Review 11.2 (1969), pp. 226–235.

[4] J. Nocedal and S.J. Wright. Numerical Optimization. Springer, 1999.

#### 4. Noise estimation

Application of the ECNoise algorithm, by Moré and Wild [5], to compute an estimate $\sigma_k$ of the noise level $\epsilon_f = Var(\epsilon)^{1/2}$ of an objective $f$:

$$\sigma_k^2 = \frac{\gamma_k}{m+1-k} \sum_{i=0}^{m-k} \Delta^k f(x_i)^2$$

where $x_i = x + ihd, i=0, ..., m$, $\gamma_k = \frac{(k!)^2}{(2k)!}$ and:

$$
\left\lbrace
\begin{array}{ll}
\Delta^0 f(x) = f(x) & \newline
\Delta^{k+1} f(x) = \Delta^k f(x+hd) - \Delta^k f(x) &
\end{array}
\right.
$$

[5] Jorge Moré and Stefan Wild. “Estimating Derivatives of Noisy Simulations”. In: ACM Transactions on Mathematical Software (TOMS) 38 (Apr. 2012).

#### 5. Hard-to-sample distribution sampling

Some distributions are easy to evaluate (the density), but hard to sample. For instance:
- multivariate Gaussian distribution with a complex covariance matrix
- posterior distribution in Bayesian inference
- simulated annealing

##### a. Markov Chain Monte Carlo (MCMC)

##### b. Importance sampling (IS)

Let us have X ~ p(x). We want to evaluate the expectation of some function f: $E_p(f(X)) = \int f(x) p(x) dx$. If X takes only discrete values, then $E_p(f(X)) = \sum f(x) p(x)$. In the context of a Monte Carlo (MC) simulation, $E_p(f(X)) \approx \frac{1}{N} \sum_{i=1}^N f(x_i)$ where the MC samples $x_i$ are drawn from distribution p(x).

However, the MC simulation under distribution p(x) may generate very few samples (or even none) in the area of interest to evaluate $E_p(f(X))$. In that case, the variance of the MC estimator may be high. To solve this issue, one can resort to Importance Sampling (IS). Instead of sampling from distribution p(x), one will sample from another distribution q(x) that will be more suitable to sample in this zone of interest. In other words, we have $E_p(f(X)) = \int f(x) \frac{p(x)}{q(x)} q(x) dx = E_q(f(X) \frac{p(X)}{q(X)}$. In a MC setting, this leads to:

$$E_p(f(X)) = \approx \frac{1}{N} \sum_{i=1}^N f(x_i) w(x_i)$$

where the MC samples $x_i$ are drawn from distribution q(x) and $w(x_i) = \frac{p(x_i)}{q(x_i)}$ are called the sampling ratios or sampling weights. These are essential to make this estimator unbiased. It is important for q(x) to be large where |f(x)p(x)| is large, so that $Var_q(f(x)\frac{p(x)}{q(x)} < Var_p(f(x))$.

###### Example:

We want to estimate the probability that a fire starts. Let us assume that a fire starts only when the temperature reaches 80 degrees celsius. But this is a rare event: most of the time, the temperature is much lower and follows e.g. a normal distribution p(x) of mean=25.0 and std=5.0. So we want to compute $p_0 = P(X>80) = \int f(x) p(x) dx$ with X ~ p(x) and $f(x) = \mathbb{1}_{X>80}$. But, in that case, a basic MC sampling will produce very few instances of temperatures above 80 degrees and the variance, based on these few examples, will be relatively large.

Instead, one can apply an Importance Sampling strategy, using a proposal distribution q(x) which is Gaussian of mean=90.0 and std=10.0. But, in a general setting, we might be unsure about the right proposal distribution q(x) to choose. One solution is Adaptive IS, which amounts to choosing a first not-so-good distribution and making it evolve. For instance, a Gaussian proposal distribution of mean=50.0 and std=10.0, whose mean is updated each X iterations, based on the sampled above 80 degrees.

##### c. Variational Inference

<br>

### Publications

List of conferences:
- G. Caron, A. Juditsky, N. Guitard, D. Céli, *Recovery of Intrinsic Heterojunction Bipolar Transistors Profiles by Neural Networks*, BCICTS 2022
- G. Caron, A. Juditsky, N. Guitard, D. Céli, *Physics-based LSTM Neural Networks Surrogate Model for SiGe HBT Intrinsic Profile Optimization*, SSDM 2024

List of inventions:
- *Failure-robust BFGS algorithm*
- *Acceleration of optimization on a neural network surrogate model* (STMicroelectronics Trade-Secret)

### Contact
E-mail: g.caron1789@gmail.com  
LinkedIn: www.linkedin.com/in/gregoire-caron


