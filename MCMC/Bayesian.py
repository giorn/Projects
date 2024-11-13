""" 
Author: Gregoire Caron 
Date Created: 2024-11-13
Last Modified: 2024-11-13
Small example of application of the Markov Chain Monte-Carlo (MCMC) Metropolis-Hastings algorithm.
Use of the emcee library for MCMC.
Goal is to find the pdf of parameters (a,b,c) for quadratic fit over data, i.e. 3-dimensional pdf.
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
plt.style.use('tableau-colorblind10')
import corner

np.random.seed(23)


def get_data(n_points, true_a, true_b, true_c):
    """Generate (x,y) data to fit."""
    x = np.linspace(-10, 10, n_points)
    y = true_a + true_b * x + true_c * x**2
    y += np.random.normal(0, 5, size=n_points)
    return x, y

class Log_probability():

    def __init__(self, guesses, prior="uniform"):
        self.mu_a, self.mu_b, self.mu_c = guesses
        self.prior = prior

    def log_likelihood(self, theta, x, y, yerr):
        """Return the log likelihood of observing the data given the parameters."""
        a, b, c = theta
        model = a + b * x + c * x**2
        return -0.5 * np.sum(((y - model) / yerr) ** 2) 

    def bound_log_prior(self, theta):
        """Return a log prior = assumption on the distribution before observing the data
        Here that parameters are within reasonable ranges.
        """
        a, b, c = theta
        if -10.0 < a < 10.0 and -5.0 < b < 5.0 and -1.0 < c < 1.0:
            return 0.0
        return -np.inf

    def gaussian_log_prior(self, theta, mu_a, mu_b, mu_c):
        """Return a gaussian log prior."""
        a, b, c = theta
        sigma_a, sigma_b, sigma_c = 1.0, 1.0, 1.0
        log_prior_a = -0.5 * ((a - mu_a) / sigma_a) ** 2 - np.log(sigma_a * np.sqrt(2 * np.pi))
        log_prior_b = -0.5 * ((b - mu_b) / sigma_b) ** 2 - np.log(sigma_b * np.sqrt(2 * np.pi))
        log_prior_c = -0.5 * ((c - mu_c) / sigma_c) ** 2 - np.log(sigma_c * np.sqrt(2 * np.pi))
        return log_prior_a + log_prior_b + log_prior_c

    def get_func(self, theta, x, y, yerr):
        """Compute the log probability using Bayes theorem."""
        if self.prior == "uniform":
            lp = self.bound_log_prior(theta)
        elif self.prior == "gaussian":
            lp = self.gaussian_log_prior(theta, self.mu_a, self.mu_b, self.mu_c)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)  # Bayes theorem in log way
    
def trace_plot(samples):
    """Show trace plots."""
    fig, ax = plt.subplots(3, figsize=(10, 7), sharex=True)
    labels = ["a", "b", "c"]
    for i in range(3):
        ax[i].plot(samples[:, i], alpha=0.5)
        ax[i].set_ylabel(labels[i])
        ax[i].set_xlabel("Step number")
    plt.show()

def autocorrelation(sampler):
    try:
        tau = sampler.get_autocorr_time()
        print("Autocorrelation times:", tau)
    except RuntimeError:
        print("Autocorrelation time estimation failed. Try running the sampler for more steps.")

def main():
    """Main function for Bayesian analysis on quadratic regression."""

    # Get data to fit
    n_points = 500
    true_a, true_b, true_c = 2.5, -1.3, 0.5
    x, y = get_data(n_points, true_a, true_b, true_c)

    # MCMC parameters
    yerr = 5  # Data std uncertainty
    n_walkers = 32  # Number of Markov chains - different initial conditions
    n_steps = 10_000
    initial_guesses = np.array([2.0, -1.0, 0.4])
    dim = 3
    pos = initial_guesses + 1e-4 * np.random.randn(n_walkers, dim)  # Initial conditions
    # MCMC initialization and launch
    log_proba = Log_probability(initial_guesses, prior="gaussian")
    sampler = emcee.EnsembleSampler(n_walkers, dim, log_proba.get_func, args=(x, y, yerr))
    sampler.run_mcmc(pos, n_steps, progress=True)
    burn_in = 1_000  # Burn-in (initial steps) to discard
    samples = sampler.get_chain(discard=burn_in, thin=15, flat=True)  # thin = frequency to save samples

    # Convergence diagnostic
    trace_plot(samples)
    autocorrelation(sampler)
    # MCMC results
    corner.corner(samples, labels=["a", "b", "c"], truths=[true_a, true_b, true_c])
    plt.show()
    # Estimated parameters
    a_mcmc, b_mcmc, c_mcmc = np.median(samples, axis=0)
    print(f"Estimated parameters: a = {a_mcmc:.3f}, b = {b_mcmc:.3f}, c = {c_mcmc:.3f}")

if __name__ == "__main__":
    main()
