""" 
Author: Gregoire Caron 
Date Created: 2024-11-04
Last Modified: 2024-11-08
Module to apply importance sampling Monte Carlo to an actual case.
This actual case is fire detection when temperature > threhold = 80.
We assume the temperature follows a distribution p(x).
If X is a random variable that follows p(x), it amounts to computing p = P(X>80).
It amounts to p = ∫f(x)p(x)dx with f the indicator function f(x) = 1 if x>80, 0 else.
It is transformed into p = ∫f(x)q(x)w(x)dx with the weight w(x) = p(x)/q(x).
q(x) is chosen to approximate f(x)p(x). This decreases the variance of the estimate of p.
"""

import numpy as np
from scipy.stats import norm, expon
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Estimation():

    def __init__(self, threshold, mean, std, n_simu):
        self.threshold = threshold
        self.mean = mean
        self.std = std
        self.n_simu = n_simu
        self.true_proba = norm.sf(self.threshold, loc=self.mean, scale=self.std)  # Survival function

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
    
    def sample_proposal(self, distribution, params):
        """
        Sample from a proposal distribution and calculate its PDF.
        
        Args:
            distribution (str): 'normal' or 'exponential'.
            params (dict): Parameters for the distribution.

        Returns:
            samples (np.ndarray): Samples from the proposal distribution.
            pdf (np.ndarray): PDF values of the samples for the proposal distribution.
        """
        if distribution == "normal":
            mean, std = params.get("mean"), params.get("std")
            samples = np.random.normal(mean, std, self.n_simu)
            pdf = norm.pdf(samples, mean, std)
        elif distribution == "exponential":
            threshold, rate = params.get("threshold"), params.get("rate")
            samples = np.random.exponential(scale=1/rate, size=self.n_simu) + threshold
            pdf = expon.pdf(samples - threshold, scale=1/rate)
        else:
            raise ValueError("Unsupported distribution type. Choose 'normal' or 'exponential'.")
        return samples, pdf
    
    def importance_sampling_MC(self, distribution, params):
        """Importance sampling Monte Carlo technique for more accurate probability estimation."""
        proposal_temp, proposal_pdf = self.sample_proposal(distribution, params)
        temp_pdf = norm.pdf(proposal_temp, self.mean, self.std)
        log_weights = np.log(temp_pdf) - np.log(proposal_pdf)
        weights = np.exp(log_weights)  # Avoid underflow / overflow
        prob_event = np.mean((proposal_temp > self.threshold).astype(float) * weights)  # IS probability estimation
        return prob_event

    def adaptive_importance_sampling_MC(self, distribution, params, nb_iterations=10):
        """Adaptive importance sampling Monte Carlo, with update via maximum likelihood estimation."""
        results = []
        for iteration in range(nb_iterations):
            proposal_temp, proposal_pdf = self.sample_proposal(distribution, params)
            temp_pdf = norm.pdf(proposal_temp, self.mean, self.std)
            log_weights = np.log(temp_pdf) - np.log(proposal_pdf)
            weights = np.exp(log_weights)  # Avoid underflow / overflow
            prob_event = np.mean((proposal_temp > self.threshold).astype(float) * weights)  # IS probability estimation
            results.append(prob_event)
            
            # Proposal distribution adaptation
            if distribution == "normal":
                logging.info(f"Iteration {iteration + 1}/{nb_iterations}, \
                        importance_mean = {params["mean"]:.2f}, prob_event = {prob_event:.4e}")
                params["mean"] = np.mean(proposal_temp[proposal_temp > self.threshold])
            elif distribution == "exponential":
                logging.info(f"Iteration {iteration + 1}/{nb_iterations}, \
                        importance_rate = {params["rate"]:.2f}, prob_event = {prob_event:.4e}")
                params["rate"] = 1 / np.mean(proposal_temp[proposal_temp > self.threshold] - params["threshold"])
        return np.mean(results)
    
def main(experiment):

    # Basic estimation
    basic_prob_event = experiment.basic_prob_estimation()
    print(f"Basic estimation method: probability of fire = {basic_prob_event}")

    # IS - Gaussian
    params_normal = {"mean": 90, "std": 10}
    importance_sampling_prob_event = \
        experiment.importance_sampling_MC(distribution="normal", params=params_normal)
    print(f"IS - Gaussian proposal: probability of fire = {importance_sampling_prob_event}")

    # IS - Exponential
    params_expo = {"threshold": 80, "rate": 0.1}
    importance_sampling_prob_event = \
        experiment.importance_sampling_MC(distribution="exponential", params=params_expo)
    print(f"IS - Exponential proposal: probability of fire = {importance_sampling_prob_event}")

    # AIS - Gaussian
    params_normal = {"mean": 50, "std": 10}
    importance_sampling_prob_event = \
        experiment.adaptive_importance_sampling_MC(distribution="normal", params=params_normal, nb_iterations=100)
    print(f"AIS - Gaussian proposal: probability of fire = {np.mean(importance_sampling_prob_event)}")

    # AIS - Exponential
    params_expo = {"threshold": 80, "rate": 0.1}
    importance_sampling_prob_event = \
        experiment.adaptive_importance_sampling_MC(distribution="exponential", params=params_expo, nb_iterations=100)
    print(f"AIS - Exponential proposal: probability of fire = {np.mean(importance_sampling_prob_event)}")

if __name__ == "__main__":

    n_simu = 100_000
    threshold = 80  # Temperature threshold for a fire to start
    mean_temp = 25
    std_dev = 5
    estim = Estimation(threshold, mean_temp, std_dev, n_simu)
    print(f"True probability of fire = {estim.true_proba}")
    main(estim)
