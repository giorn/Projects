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

from Importance_sampling.main import Estimation

np.random.seed(0)


@pytest.fixture
def experiment():
    """Fixture to set up an instance of Estimation for testing."""
    n_simu = 100_000
    threshold = 80
    mean_temp = 25
    std_dev = 5
    return Estimation(threshold, mean_temp, std_dev, n_simu)

def test_fire_proba_estimation(experiment):
    """Test that the fire probability estimate is as expected."""
    basic_prob_event = experiment.basic_prob_estimation()
    importance_mean = 50
    importance_std = 10
    nb_iterations = 100
    importance_sampling_prob_event = \
        np.mean(experiment.adaptive_importance_sampling_MC(importance_mean, importance_std, nb_iterations))
    assert importance_sampling_prob_event == pytest.approx(basic_prob_event, rel=0.10)
