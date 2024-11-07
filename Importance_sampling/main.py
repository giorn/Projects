""" 
Author: Gregoire Caron 
Date Created: 2024-11-04
Last Modified: 2024-11-07
Module to apply importance sampling Monte Carlo to an actual case.
This actual case is fire detection when temperature > threhold = 80.
We assume the temperature follows a distribution p(x).
If X is a random variable that follows p(x), it amounts to computing p = P(X>80).
It amounts to p = ∫f(x)p(x)dx with f the indicator function f(x) = 1 if x>80, 0 else.
It is transformed into p = ∫f(x)q(x)w(x)dx with the weight w(x) = p(x)/q(x).
q(x) is chosen to approximate f(x)p(x). This decreases the variance of the estimate of p.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class Estimation():

    def __init__(self, threshold, mean, std, n_simu):
        self.threshold = threshold
        self.mean = mean
        self.std = std
        self.n_simu = n_simu

    def get_temp(self):
        """Return temperatures sampled from a normal distribution."""
        return np.random.normal(self.mean, self.std, self.n_simu)
    
    def plot_temp(self, temp):
        """Plot temperature distribution."""
        plt.hist(temp, bins=50, density=True, alpha=0.6, color='g')
        plt.xlabel("Temperature")
        plt.ylabel("Probability density")
        plt.grid()
        plt.tight_layout()
        plt.show()

    def basic_prob_estimation(self, plot=False):
        """Basic (and highly inaccurate) way to estimate the probability."""
        # Normal distribution for the temperatures
        temp = self.get_temp()
        if plot:
            self.plot_temp(temp)
        # Estimation of the probability for a fire to start
        prob_event = np.mean(temp > self.threshold)
        return prob_event
    
    def importance_sampling_MC(self, importance_mean, importance_std):
        """Importance sampling Monte Carlo technique for more accurate probability estimation."""
        proposal_temp = np.random.normal(importance_mean, importance_std, self.n_simu)
        temp_density = norm.pdf(proposal_temp, self.mean, self.std)  # Densité de probabilité de la distribution cible
        new_temp_density = norm.pdf(proposal_temp, importance_mean, importance_std)
        weights = (temp_density / new_temp_density)
        prob_event = np.mean((proposal_temp > self.threshold).astype(float) * weights)
        return prob_event

    def adaptive_importance_sampling_MC(self, importance_mean, importance_std, nb_iterations):
        """Adaptive importance sampling Monte Carlo, with update via maximum likelihood estimation."""
        results = []
        for iteration in range(nb_iterations):
            proposal_temp = np.random.normal(importance_mean, importance_std, self.n_simu)
            temp_density = norm.pdf(proposal_temp, self.mean, self.std)
            new_temp_density = norm.pdf(proposal_temp, importance_mean, importance_std)
            weights = (temp_density / new_temp_density)
            prob_event = np.mean((proposal_temp > self.threshold).astype(float) * weights)
            results.append(prob_event)
            importance_mean = np.mean(proposal_temp[proposal_temp > self.threshold])
        return results

if __name__ == "__main__":

    n_simu = 100_000
    threshold = 80  # Temperature threshold for a fire to start
    mean_temp = 25
    std_dev = 5

    # Basic estimation
    estim = Estimation(threshold, mean_temp, std_dev, n_simu)
    basic_prob_event = estim.basic_prob_estimation()
    print(f"Basic (and highly inaccurate) estimation method: probability of fire = {basic_prob_event}")

    # Importance sampling estimation
    importance_mean = 90
    importance_std = 10
    importance_sampling_prob_event = \
        estim.importance_sampling_MC(importance_mean, importance_std)
    print(f"Importance sampling Monte Carlo: probability of fire = {importance_sampling_prob_event}")

    # Adaptive importance sampling estimation
    importance_mean = 50
    importance_std = 10
    nb_iterations = 100
    importance_sampling_prob_event = \
        estim.adaptive_importance_sampling_MC(importance_mean, importance_std, nb_iterations)
    print(f"Importance sampling Monte Carlo: probability of fire = {np.mean(importance_sampling_prob_event)}")
