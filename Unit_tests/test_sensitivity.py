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
    data = np.random.rand(100)*2*np.pi
    data = data.reshape(-1, 1)
    return Sensitivity(model, scaler, data)

def test_data_shape(experiment):
    """Test that the derivatives are the ones expected."""
    experiment.compute_gradient()
    mu, sigma, mu_sqd = experiment.get_measures()
    print(mu, sigma)
    assert mu == pytest.approx(1, rel=0.05)
    assert sigma == pytest.approx(np.sqrt(2)/2, rel=0.05)
