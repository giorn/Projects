""" 
Author: Gregoire Caron 
Date Created: 2024-11-07
Last Modified: 2024-11-08
Same case as in main.py but with a more complex distribution of temperatures (no analytical expression).
"""

import numpy as np
from scipy.stats import norm, gaussian_kde
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import os

os.environ['OMP_NUM_THREADS'] = '1'


class Estimation():

    def __init__(self, threshold, mean, std, n_simu):
        self.threshold = threshold
        self.mean = mean
        self.std = std
        self.n_simu = n_simu

    def get_temp(self):
        """Return temperatures sampled from a normal distribution."""
        return np.random.normal(self.mean, self.std, self.n_simu)
    
    def get_complex_temp(self):
        """Return temperatures sampled from a mixture of Gaussians."""
        means = [10, 25, 35, 45]
        std_devs = [3, 5, 10, 7]
        weights = [0.1, 0.3, 0.4, 0.2]
        temp = np.zeros(self.n_simu)
        for i in range(self.n_simu):
            component = np.random.choice(len(means), p=weights)
            temp[i] = np.random.normal(means[component], std_devs[component])
        prob_above_threshold = sum(w * (1 - norm.cdf(self.threshold, loc=mu, scale=std))
                            for w, mu, std in zip(weights, means, std_devs))
        # This probability is always exact and stays the same (for a given threshold value)
        print(f"Analytical probability to be above {self.threshold}: {prob_above_threshold}")
        return temp

    def plot_temp(self):
        """Plot temperature distribution."""
        plt.hist(self.temp, bins=50, density=True, alpha=0.6, color='g')
        plt.xlabel("Temperature")
        plt.ylabel("Probability density")
        plt.grid()
        plt.tight_layout()
        plt.show()

    def get_KDE(self, data, plot=False):
        """Fit KDE on basic temperatures and evaluate density for given input data."""
        # This fit varies from one run to another, especially on the tails (few points)
        # The values p(x) to compute the weights p(x)/q(x) may vary by several orders of magnitude
        # Therefore, the probability estimate p may also vary greatly
        kde = gaussian_kde(self.temp, bw_method='scott')
        x = np.arange(0, 100, 0.01)
        if plot:
            plt.hist(self.temp, bins=30, density=True, alpha=0.5, color='skyblue')
            plt.plot(x, kde(x))
            plt.show()
        kde_values = kde(data)
        return kde_values
    
    def find_best_GMM(self, plot=False):
        """Find the best number of components for the GMM."""
        lowest_bic = np.infty
        bic = []
        n_components_range = range(1, 20)
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=23)
            gmm.fit(self.temp.reshape(-1, 1))
            bic.append(gmm.bic(self.temp.reshape(-1, 1)))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
        plt.plot(n_components_range, bic, label="BIC")
        plt.legend()
        plt.show()
        return best_gmm
    
    def get_GMM(self, data, plot=False):
        """Fit KDE on basic temperatures and evaluate density for given input data."""
        n_components = 4
        gmm = self.find_best_GMM()
        x = np.arange(0, 100, 0.01).reshape(-1, 1)
        if plot:
            plt.hist(self.temp, bins=30, density=True, alpha=0.5, color='skyblue')
            logprob = gmm.score_samples(x)
            pdf = np.exp(logprob)
            plt.plot(x, pdf)
            plt.show()
        logprob = gmm.score_samples(data.reshape(-1, 1))
        gmm_values = np.exp(logprob)
        return gmm_values

    def basic_prob_estimation(self, plot=False):
        """Basic (and highly inaccurate) way to estimate the probability."""
        # Normal distribution for the temperatures
        self.temp = self.get_complex_temp()
        if plot:
            self.plot_temp()
        # Estimation of the probability for a fire to start
        prob_event = np.mean(self.temp > self.threshold)
        return prob_event
    
    def importance_sampling_MC(self, importance_mean, importance_std, density_estim_method="GMM"):
        """Importance sampling Monte Carlo technique for more accurate probability estimation."""
        proposal_temp = np.random.normal(importance_mean, importance_std, self.n_simu)
        if density_estim_method == "KDE":
            temp_density = self.get_KDE(proposal_temp, plot=False)
        else:
            temp_density = self.get_GMM(proposal_temp, plot=False)
        new_temp_density = norm.pdf(proposal_temp, importance_mean, importance_std)
        weights = (temp_density / new_temp_density)
        prob_event = np.mean((proposal_temp > self.threshold).astype(float) * weights)
        return prob_event

if __name__ == "__main__":

    n_simu = 10_000
    threshold = 80  # Temperature threshold for a fire to start
    mean_temp = 25
    std_dev = 5

    # Basic estimation
    estim = Estimation(threshold, mean_temp, std_dev, n_simu)
    basic_prob_event = estim.basic_prob_estimation()
    print(f"Basic (and more inaccurate) estimation method: probability of fire = {basic_prob_event}")

    # Importance sampling estimation
    importance_mean = 80
    importance_std = 10
    # KDE much slower than GMM ofc
    # Good illustration that it is difficult to estimate p(x) correctly enough to then compute the weights and the IS proba
    importance_sampling_prob_event = \
        estim.importance_sampling_MC(importance_mean, importance_std, density_estim_method="GMM")
    print(f"Importance sampling Monte Carlo: probability of fire = {importance_sampling_prob_event}")
