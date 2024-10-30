import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import Benchmark_Functions.pybenchfunction as bench


class Failure():

    def __init__(self, func, proba_of_failure=0.00):
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
        
def main(nb_of_opti, func, fun_tol=1e-4, method="BFGS", proba_of_failure=0.00, verbose=0):
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
            opti_results_list.append(-np.nan)

    return np.array(opti_results_list)

def get_performance_profile(results_dic):
    """Plot performance profiles for given sets of optimizations."""
    global_min = min(np.nanmin(l) for l in results_dic.values())
    print(global_min)
    for proba, results_list in results_dic.items():
        full_length = len(results_list)
        results_list = results_list[~np.isnan(results_list)] / global_min
        results_list = results_list 
        results_list = np.sort(results_list)
        probas_cumulees = np.arange(1, len(results_list) + 1) / full_length
        plt.step(results_list, probas_cumulees, where="post", label=f"{proba} %")
    plt.xlabel(r"$NF / NF_{min}$")
    plt.ylabel("Proportion of convergence")
    plt.legend(title="Probability of failure")
    plt.grid()
    plt.tight_layout()
    plt.show()
        
if __name__ == "__main__":

    # Set up benchmark function
    dim = 3
    func = bench.function.Rosenbrock(dim)
    X_min, minimum = func.get_global_minimum(dim)

    # Optimization parameters
    fun_tol = 1e-4
    method = "BFGS"
    
    # Launch several optimizations
    nb_of_opti = 100
    opti_list_000 = main(nb_of_opti, func, fun_tol, method, proba_of_failure=0.00)
    opti_list_001 = main(nb_of_opti, func, fun_tol, method, proba_of_failure=0.01)
    get_performance_profile({0.00:opti_list_000, 0.01:opti_list_001})
