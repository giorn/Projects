""" 
Author: Gregoire Caron 
Date Created: 2024-10-31
Last Modified: 2024-10-31
Module displaying an application of the ECNoise algorithm by More and Wild.
"""

import numpy as np
import matplotlib.pyplot as plt


def noisy_function(x, noise_std=0.1):
    """Defines a noisy function."""
    smooth = x*x
    noise = np.random.normal(0, noise_std, size=np.shape(x))
    return smooth + noise

def plot_function(x, y):
    """Displays a function (x,y)"""
    plt.plot(x, y)
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    x = np.linspace(-1, 1, 1_000)
    y = noisy_function(x, noise_std=0.01)
    plot_function(x, y)
