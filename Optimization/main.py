import numpy as np
from scipy.optimize import minimize

import Benchmark_Functions.pybenchfunction as bench


class Failure():

    def __init__(self, func, failure_proba):
        self.func = func
        self.failure_proba = failure_proba
    
    def function(self, x):
        """Modified benchmark function with a probability of failure."""
        unif_nb = np.random.uniform(0, 1)
        if unif_nb <= self.failure_proba:
            print("Crash")
            return "Crash"
        else:
            print("here")
            return self.func(x)

# Set up benchmark function
dim = 3
func = bench.function.Rosenbrock(dim)
X_min, minimum = func.get_global_minimum(dim)

# Optimization
x0 = [0.0, 2.0, 0.0]
method = "BFGS"
fail = Failure(func, 0.01)
res = minimize(fail.function, x0, method=method)
print(res.x)
