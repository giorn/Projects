""" 
Author: Gregoire Caron 
Date Created: 2024-10-31
Last Modified: 2024-11-01
Module displaying an application of the ECNoise algorithm by More and Wild.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def noisy_function(x, noise_std=0.01):
    """Define a noisy function."""
    smooth = x*x
    noise = np.random.normal(0, noise_std, size=np.shape(x))
    return smooth + noise

def plot_function(x, y):
    """Display a function (x,y)"""
    plt.plot(x, y)
    plt.grid()
    plt.tight_layout()
    plt.show()

def has_both_negative_and_positive(nb_list):
    """Check if a given list of numbers contains both negative and positive numbers."""
    has_positive = any(x > 0 for x in nb_list)
    has_negative = any(x < 0 for x in nb_list)
    return has_positive and has_negative

def noise_estimation(x, h, m):
    """Compute estimate sigma_k of the noise level."""
    pts = np.arange(x-(m//2)*h, x+(m//2)*h, h)
    delta_k = noisy_function(pts, noise_std=0.01)
    #delta_k = [1.00327, 1.01081, 1.0205, 1.0325, 1.04117, 1.04955, 1.05907]
    delta_list, sigma_list = [], []
    for k in range(m):
        delta_k = np.diff(delta_k)
        gamma_k = (math.factorial((k+1))**2)/math.factorial(2*(k+1))
        sigma_k = np.sqrt((gamma_k/(m+1-(k+1)))*np.sum(delta_k**2))
        delta_list.append(has_both_negative_and_positive(delta_k))
        sigma_list.append(sigma_k)

    for l in range(len(sigma_list)-2):
        conv_condition = (np.amax(sigma_list[l:l+2] <= 4*np.amin(sigma_list[l:l+2])))
        noise_contamination = delta_list[l]
        if conv_condition and noise_contamination:
            return sigma_list[l]

if __name__ == "__main__":
    noise_estimate = noise_estimation(x=1, h=1e-3, m=10)
    print(noise_estimate)
