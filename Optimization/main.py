import numpy as np
from scipy.optimize import minimize

import Benchmark_Functions.pybenchfunction as bench


class Trigger(Exception):
    pass

class ObjectiveFunctionWrapper:

    def __init__(self, fun, fun_tol=None):
        self.fun = fun
        self.fun_tol = fun_tol
        self.last_x = 0
        self.last_f = 0
        self.number_of_f_evals = 0
        self.number_of_iter = 0

    def __call__(self, x):
        print("call")
        self.last_x = x
        self.last_f = self.fun(x)
        print(self.last_f)
        self.number_of_f_evals += 1
        return self.last_f

    def stop(self, *args):
        print("stop")
        self.number_of_iter += 1
        if self.last_f < self.fun_tol:
            raise Trigger

class Failure():

    def __init__(self, func, failure_proba):
        self.func = func
        self.failure_proba = failure_proba
    
    def function(self, x):
        """Modified benchmark function with a probability of failure."""
        unif_nb = np.random.uniform(0, 1)
        if unif_nb <= self.failure_proba:
            return "Crash"
        else:
            return self.func(x)

# Set up benchmark function
dim = 3
func = bench.function.Rosenbrock(dim)
X_min, minimum = func.get_global_minimum(dim)

# Optimization
x0 = [0.0, 2.0, 0.0]
fun_tol = 1e-4
failure_proba = 0.05
method = "BFGS"
fail = Failure(func, failure_proba)
f_wrapped = ObjectiveFunctionWrapper(fail.function, fun_tol)

try:
    res = minimize(f_wrapped, x0, method=method, callback=f_wrapped.stop)
except Trigger:
    print(f"{f_wrapped.number_of_f_evals} f-evals")
    print(f"{f_wrapped.number_of_iter} iterations")
    print(f"x = {f_wrapped.last_x}")
    print(f"f(x) = {f_wrapped.last_f}")
except Exception as e:
    raise e
