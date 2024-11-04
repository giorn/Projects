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


class Estimation():

    def __init__(self, threshold, mean_temp, std_dev, n_simu):
        self.threshold = threshold
        self.mean_temp = mean_temp
        self.std_dev = std_dev
        self.n_simu = n_simu

    def basic_prob_estimation(self):
        """Basic (and highly inaccurate) way to estimate the probability."""
        # Normal distribution for the temperatures
        temperatures = np.random.normal(self.mean_temp, self.std_dev, self.n_simu)
        # Estimation of the probability for a fire to start
        prob_event = np.mean(temperatures > self.threshold)
        return prob_event
    
    def importance_sampling_MC_estimation(self, new_mean, new_std):
        """Importance sampling Monte Carlo technique for more accurate probability estimation."""
        proposal_temperatures = np.random.normal(new_mean, new_std, self.n_simu)
        prob_event = np.mean(proposal_temperatures > self.threshold) * 1

if __name__ == "__main__":

    n_simu = 100000
    threshold = 80  # Temperature threshold for a fire to start
    mean_temp = 25
    std_dev = 5

    estim = Estimation(threshold, mean_temp, std_dev, n_simu)
    prob_event = estim.basic_prob_estimation()
    print(prob_event)
