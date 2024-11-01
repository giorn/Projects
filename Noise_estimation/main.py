""" 
Author: Gregoire Caron 
Date Created: 2024-10-31
Last Modified: 2024-10-31
Module displaying an application of the ECNoise algorithm by More and Wild.
"""

import numpy as np
import matplotlib.pyplot as plt
import math


def noisy_function(x, noise_std=0.1):
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

def noise_estimation(x, h, m):
    """Compute estimate sigma_k of the noise level."""
    pts = np.arange(x-(m//2)*h, x+(m//2)*h, h)
    delta_f = noisy_function(pts, noise_std=0.01)
    #delta_k = [1.00327, 1.01081, 1.0205, 1.0325, 1.04117, 1.04955, 1.05907]
    for k in range(m):
        delta_k = np.diff(delta_k)
        gamma_k = (math.factorial((k+1))**2)/math.factorial(2*(k+1))
        sigma_k = np.sqrt((gamma_k/(m+1-(k+1)))*np.sum(delta_k**2))
        print("{:.2e}".format(sigma_k))

if __name__ == "__main__":
    noise_estimation(x=0, h=1e-2, m=6)
