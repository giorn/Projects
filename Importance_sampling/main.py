""" 
Author: Gregoire Caron 
Date Created: 2024-11-04
Last Modified: 2024-11-04
Module to apply importance sampling Monte Carlo to an actual case.
"""

import numpy as np


class Estimation():

    def __init__(self, threshold):
        self.threshold = threshold

    def basic_prob_estimation(self, mean_temp, std_dev, n_simu):
        """Basic (and highly inaccurate) way to estimate the probability."""
        # Normal distribution for the temperatures
        temperatures = np.random.normal(mean_temp, std_dev, n_simu)
        # Estimation of the probability for a fire to start
        prob_event = np.mean(temperatures > self.threshold)
        return prob_event

if __name__ == "__main__":

    n_simu = 100000
    threshold = 80  # Temperature threshold for a fire to start
    mean_temp = 25
    std_dev = 5

    estim = Estimation(threshold)
    prob_event = estim.basic_prob_estimation(mean_temp, std_dev, n_simu)
    print(prob_event)
