import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import Benchmark_Functions.pybenchfunction as bench


# Set up benchmark function
dim = 3
func = bench.function.Rosenbrock(dim)
X_min, minimum = func.get_global_minimum(dim)
print(X_min, minimum)

# Optimization
x0 = [0.0, 2.0, 0.0]
res = minimize(func, x0, method='BFGS')
print(res.x)
