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

from Sensitivity_analysis.get_sensitivity import Sensitivity

np.random.seed(0)


@pytest.fixture
def experiment():
    """Fixture to set up an instance of Sensitivity for testing."""
    model_path = "Sensitivity_analysis/regression_model.h5"
    scaler_path = "Sensitivity_analysis/regression_scaler.gz"
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    X1 = (np.random.rand(1000)*2*np.pi).reshape(-1, 1)
    X2 = (np.random.rand(1000)).reshape(-1, 1)
    data = np.concatenate((X1, X2), axis=1)
    return Sensitivity(model, scaler, data)

def test_data_shape_00(experiment):
    """Test that the derivatives are the ones expected."""
    experiment.compute_gradient()
    mu, sigma, mu_sqd = experiment.get_measures(0, 0)
    assert mu == pytest.approx(0, abs=0.10)
    assert sigma == pytest.approx(np.sqrt(2)/2, abs=0.10)

def test_data_shape_01(experiment):
    """Test that the derivatives are the ones expected."""
    experiment.compute_gradient()
    mu, sigma, mu_sqd = experiment.get_measures(0, 1)
    assert mu == pytest.approx(0, abs=0.10)
    assert sigma == pytest.approx(3*np.sqrt(2)/2, abs=0.10)
