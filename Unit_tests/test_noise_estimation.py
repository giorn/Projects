""" 
Author: Gregoire Caron 
Date Created: 2024-10-29
Last Modified: 2024-10-31
Unit tests for sensitivity analysis.
"""

import pytest
import joblib
import tensorflow as tf
import numpy as np

from Noise_estimation.main import *


def test_noise_estimation_0_01():
    """Test that the noise estimate is approximately equal to 0.01."""
    std = 0.01
    Noise = Noisy_function(noise_std=std)
    noise_estimate = noise_estimation(f=Noise.func, x=1, h=1e-3, m=10)
    assert noise_estimate == pytest.approx(std, rel=0.10)

def test_noise_estimation_0_0005():
    """Test that the noise estimate is approximately equal to 0.01."""
    std = 0.0005
    Noise = Noisy_function(noise_std=std)
    noise_estimate = noise_estimation(f=Noise.func, x=1, h=1e-8, m=24)
    assert noise_estimate == pytest.approx(std, rel=0.10)
