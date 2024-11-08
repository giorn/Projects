""" 
Author: Gregoire Caron 
Date Created: 2024-11-08
Last Modified: 2024-11-08
Small example of application of the Markov Chain Monte-Carlo Metropolis-Hastings algorithm.
Useful when sampling a distribution using basic Monte-Carlo is too long or expensive.
"""

import numpy as np
import matplotlib.pyplot as plt


def target_distribution(x):
    """Return the probability at x for a mixture of Gaussian distributions."""
    means = [4, -2]
    stds = [0.5, 0.8]
    weights = [0.3, 0.7]
    proba = np.sum([w*np.exp(-0.5*((x-m)/s)**2) for w, m, s in zip(weights, means, stds)], axis=0)
    return proba

class Sampling():

    def __init__(self, target_dist, n_samples):
        self.target_dist = target_dist
        self.n_samples = n_samples
        self.adapt_interval = n_samples/20

    def Metropolis_Hastings(self):
        """Sample from the target distribution using the Metropolis-Hastings algorithm."""
        proposal_std = 1.0
        samples = np.zeros(self.n_samples)
        x_current = 0
        acceptance_count = 0
        for i in range(1, self.n_samples):
            x_proposal = x_current + np.random.normal(0, proposal_std)
            acceptance_ratio = target_distribution(x_proposal) / (target_distribution(x_current) + 1e-20)
            if np.random.rand() < acceptance_ratio:
                x_current = x_proposal
            samples[i] = x_current
            if i % self.adapt_interval == 0:
                print(proposal_std)
                acceptance_rate = acceptance_count / self.adapt_interval
                if acceptance_rate < 0.2:
                    proposal_std *= 1.1
                elif acceptance_rate > 0.3:
                    proposal_std *= 0.9
                acceptance_count = 0
        return samples

    def plot_distrib(self, samples):
        """"""
        x = np.linspace(-6, 6, 500)
        plt.hist(samples, bins=50, density=True, alpha=0.6, color='blue', label='Samples (MCMC)')
        plt.plot(x, self.target_dist(x), 'r', label='Target distribution')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    n_samples = 10_000
    sampling = Sampling(target_distribution, n_samples)
    samples = sampling.Metropolis_Hastings()
    sampling.plot_distrib(samples)
