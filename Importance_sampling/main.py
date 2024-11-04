""" 
Author: Gregoire Caron 
Date Created: 2024-11-04
Last Modified: 2024-11-04
Module to apply importance sampling Monte Carlo to an actual case.
This actual case is fire detection when temperature > threhold = 80.
The goal is to compute the integral (mean) G, over [0,100], of a function g(x).
g is an indicator function (0 if x < 80, 1 else).
"""

import numpy as np
from scipy.stats import norm


class Estimation():

    def __init__(self, threshold, mean, std, n_simu):
        self.threshold = threshold
        self.mean = mean
        self.std = std
        self.n_simu = n_simu

    def basic_prob_estimation(self):
        """Basic (and highly inaccurate) way to estimate the probability."""
        # Normal distribution for the temperatures
        temperatures = np.random.normal(self.mean, self.std, self.n_simu)
        # Estimation of the probability for a fire to start
        prob_event = np.mean(temperatures > self.threshold).astype(float)
        return prob_event
    
    def importance_sampling_MC(self, importance_mean, importance_std):
        """Importance sampling Monte Carlo technique for more accurate probability estimation."""
        proposal_temperatures = np.random.normal(importance_mean, importance_std, self.n_simu)
        temp_density = norm.pdf(proposal_temperatures, self.mean, self.std)  # Densité de probabilité de la distribution cible
        new_temp_density = norm.pdf(proposal_temperatures, importance_mean, importance_std)
        prob_event = np.mean((proposal_temperatures > self.threshold).astype(float) * (temp_density / new_temp_density))
        return prob_event

    def adaptive_importance_sampling_MC(self, importance_mean, importance_std):
        """Adaptive importance sampling Monte Carlo, with update via maximum likelihood estimation."""
        results = []
        for iteration in range(10):
            proposal_temperatures = np.random.normal(importance_mean, importance_std, self.n_simu)
            temp_density = norm.pdf(proposal_temperatures, self.mean, self.std)
            new_temp_density = norm.pdf(proposal_temperatures, importance_mean, importance_std)
            prob_event = np.mean((proposal_temperatures > self.threshold).astype(float) * (temp_density / new_temp_density))
            results.append(prob_event)
            importance_mean = np.mean(proposal_temperatures)
            importance_std = np.std(proposal_temperatures)
            print(importance_mean)
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
    importance_sampling_prob_event = \
        estim.adaptive_importance_sampling_MC(importance_mean, importance_std)
    print(f"Importance sampling Monte Carlo: probability of fire = {importance_sampling_prob_event}")
