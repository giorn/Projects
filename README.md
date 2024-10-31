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

Application of the ECNoise algorithm, by Moré and Wild [5].

[5] Jorge Moré and Stefan Wild. “Estimating Derivatives of Noisy Simulations”. In: ACM Transactions on Mathematical Software (TOMS) 38 (Apr. 2012). 

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
LinkedIn: www.linkedin.com/in/grégoire-caron-b27573196
