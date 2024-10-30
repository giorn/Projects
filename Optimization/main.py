import numpy as np
from scipy.optimize import minimize

import Benchmark_Functions.pybenchfunction as bench


class Failure():

    def __init__(self, func, proba_of_failure):
        self.func = func
        self.proba_of_failure = proba_of_failure
    
    def function(self, x):
        """Modified benchmark function with a probability of failure."""
        unif_nb = np.random.uniform(0, 1)
        if unif_nb <= self.proba_of_failure:
            return "Crash"
        else:
            return self.func(x)

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
        """Calls to wrapped function."""
        self.last_x = x
        self.last_f = self.fun(x)
        self.number_of_f_evals += 1
        return self.last_f

    def stop(self, *args):
        """Callback: checks if termination condition is reached."""
        self.number_of_iter += 1
        if self.last_f < self.fun_tol:
            raise Trigger
        
def main(nb_of_opti, func, proba_of_failure, fun_tol, method, verbose=0):
    """Launches several optimizations.
    Inputs:
    - nb_of_opti: number of optimizations launched
    - func: function to optimize
    - proba_of_failure: probability of failure
    - fun_tol: termination criterion
    - method: optimization scheme
    - verbose: print (1) or not (0) the results of the optimization
    """
    fail = Failure(func, proba_of_failure)
    opti_results_list = []

    for i in range(nb_of_opti):
        x0 = np.random.rand(3)*2
        f_wrapped = ObjectiveFunctionWrapper(fail.function, fun_tol)
        try:
            res = minimize(f_wrapped, x0, method=method, callback=f_wrapped.stop)
        except Trigger:
            if verbose:
                print(f"{f_wrapped.number_of_f_evals} f-evals")
                print(f"{f_wrapped.number_of_iter} iterations")
                print(f"x = {f_wrapped.last_x}")
                print(f"f(x) = {f_wrapped.last_f}")
            opti_results_list.append(f_wrapped.number_of_iter)
        except Exception:
            if verbose:
                print("Crash")
            opti_results_list.append("Crash")

    return opti_results_list

def get_performance_profile(results_list):

        
if __name__ == "__main__":

    # Set up benchmark function
    dim = 3
    func = bench.function.Rosenbrock(dim)
    X_min, minimum = func.get_global_minimum(dim)

    # Optimization parameters
    fun_tol = 1e-4
    proba_of_failure = 0.00
    method = "BFGS"
    
    # Launch several optimizations
    nb_of_opti = 100
    opti_results_list = main(nb_of_opti, func, proba_of_failure, fun_tol, method)
    print(opti_results_list)
