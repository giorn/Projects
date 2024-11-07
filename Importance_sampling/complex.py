""" 
Author: Gregoire Caron 
Date Created: 2024-11-07
Last Modified: 2024-11-07
Same case as in main.py but with a more complex distribution of temperatures (no analytical expression).
"""

import numpy as np
from scipy.stats import norm, gaussian_kde
import matplotlib.pyplot as plt


class Estimation():

    def __init__(self, threshold, mean, std, n_simu):
        self.threshold = threshold
        self.mean = mean
        self.std = std
        self.n_simu = n_simu
    
    def get_complex_temp(self):
        """Return temperatures sampled from a mixture of Gaussians."""
        means = [10, 25, 35, 45]
        std_devs = [3, 5, 10, 7]
        weights = [0.1, 0.3, 0.4, 0.2]
        temp = np.zeros(self.n_simu)
        for i in range(self.n_simu):
            component = np.random.choice(len(means), p=weights)
            temp[i] = np.random.normal(means[component], std_devs[component])
        return temp

    def plot_temp(self):
        plt.hist(self.temp, bins=50, density=True, alpha=0.6, color='g')
        plt.xlabel("Temperature")
        plt.ylabel("Probability density")
        plt.grid()
        plt.tight_layout()
        plt.show()

    def get_KDE(self, data, plot=False):
        """Fit KDE on basic temperatures and ..."""
        kde = gaussian_kde(self.temp, bw_method='scott')
        x = np.arange(0, 100, 0.01)
        if plot:
            plt.hist(self.temp, bins=30, density=True, alpha=0.5, color='skyblue')
            plt.plot(x, kde(x))
            plt.show()
        kde_values = kde(data)
        return kde_values

    def basic_prob_estimation(self, plot=False):
        """Basic (and highly inaccurate) way to estimate the probability."""
        # Normal distribution for the temperatures
        self.temp = self.get_complex_temp()
        if plot:
            self.plot_temp()
        # Estimation of the probability for a fire to start
        prob_event = np.mean(self.temp > self.threshold)
        return prob_event
    
    def importance_sampling_MC(self, importance_mean, importance_std):
        """Importance sampling Monte Carlo technique for more accurate probability estimation."""
        proposal_temp = np.random.normal(importance_mean, importance_std, self.n_simu)
        temp_density = self.get_KDE(proposal_temp, plot=False)
        new_temp_density = norm.pdf(proposal_temp, importance_mean, importance_std)
        weights = (temp_density / new_temp_density)
        prob_event = np.mean((proposal_temp > self.threshold).astype(float) * weights)
        return prob_event

if __name__ == "__main__":

    n_simu = 100_000
    threshold = 60  # Temperature threshold for a fire to start
    mean_temp = 25
    std_dev = 5

    # Basic estimation
    estim = Estimation(threshold, mean_temp, std_dev, n_simu)
    basic_prob_event = estim.basic_prob_estimation()
    print(f"Basic (and highly inaccurate) estimation method: probability of fire = {basic_prob_event}")

    # Importance sampling estimation
    importance_mean = 60
    importance_std = 10
    importance_sampling_prob_event = \
        estim.importance_sampling_MC(importance_mean, importance_std)
    print(f"Importance sampling Monte Carlo: probability of fire = {importance_sampling_prob_event}")
