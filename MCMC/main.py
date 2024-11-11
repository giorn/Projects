""" 
Author: Gregoire Caron 
Date Created: 2024-11-08
Last Modified: 2024-11-08
Small example of application of the Markov Chain Monte-Carlo Metropolis-Hastings algorithm.
Useful when sampling a distribution using basic Monte-Carlo is too long or expensive.
[Roberts97] Optimal acceptance rate = 0.234.
"""

import numpy as np
import matplotlib.pyplot as plt


def target_distribution(x):
    """Return the probability at x for a mixture of Gaussian distributions."""
    means = [20, -2]
    stds = [0.5, 0.8]
    weights = [0.3, 0.7]
    proba = np.sum([w*np.exp(-0.5*((x-m)/s)**2) for w, m, s in zip(weights, means, stds)], axis=0)
    return proba

def autocorrelation(samples, max_lag):
    """Compute autocorrelation for each lag."""
    autocorr = np.array([np.corrcoef(samples[:-lag], samples[lag:])[0, 1] for lag in range(1, max_lag + 1)])
    return autocorr

class Sampling():

    def __init__(self, target_dist, n_samples, n_chains, low_x0, high_x0, std):
        self.target_dist = target_dist
        self.n_samples = n_samples
        self.n_chains = n_chains
        self.adapt_interval = n_samples/10
        self.low_x0 = low_x0
        self.high_x0 = high_x0
        self.std = std

    def Metropolis_Hastings(self):
        """Sample from the target distribution using the Metropolis-Hastings algorithm."""
        all_samples = []
        for nc in range(self.n_chains):
            proposal_std = self.std  # The further away the peaks, the greater the initial std should be
            samples = np.zeros(self.n_samples)
            x_current = np.random.uniform(self.low_x0, self.high_x0)
            acceptance_count = 0
            for ns in range(1, self.n_samples):
                x_proposal = x_current + np.random.normal(0, proposal_std)
                acceptance_ratio = target_distribution(x_proposal) / (target_distribution(x_current) + 1e-20)
                if np.random.rand() < acceptance_ratio:
                    x_current = x_proposal
                    acceptance_count += 1
                samples[ns] = x_current
                if ns % self.adapt_interval == 0:
                    print(proposal_std)
                    acceptance_rate = acceptance_count / self.adapt_interval
                    if acceptance_rate < 0.2:
                        proposal_std *= 0.95
                    elif acceptance_rate > 0.3:
                        proposal_std *= 1.1
                    acceptance_count = 0
            all_samples.append(samples)
        return np.array(all_samples)

    def plot_distrib(self, all_samples):
        """Plot distribution of MCMC samples."""
        all_samples_flat = all_samples.flatten()
        x = np.linspace(min(all_samples_flat) - 2, max(all_samples_flat) + 2, 500)
        plt.hist(all_samples_flat, bins=100, density=True, alpha=0.6, color='blue', label='Samples (MCMC)')
        plt.plot(x, self.target_dist(x), 'r', label='Target distribution')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    def trace_plot(self, all_samples):
        """Plot evolution of sampled values with MCMC iterations."""
        for i in range(self.n_chains):
            plt.plot(np.arange(self.n_samples), all_samples[i], label=f"{i}th chain")
        plt.xlabel('Iteration')
        plt.ylabel('x')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_autocorrelation(self, samples, max_lag=100):
        """Plot autocorrelation evolution with the lag."""
        autocorr_values = autocorrelation(samples, max_lag)
        plt.bar(range(1, max_lag + 1), autocorr_values)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    n_samples = 100_000
    n_chains = 2
    low_x0, high_x0 = -2, 20
    std = 30.0
    sampling = Sampling(target_distribution, n_samples, n_chains, low_x0, high_x0, std)
    all_samples = sampling.Metropolis_Hastings()
    sampling.plot_distrib(all_samples)
    sampling.trace_plot(all_samples)
    sampling.plot_autocorrelation(all_samples[0], max_lag=1_000)
