import pytest
import joblib
import tensorflow as tf

from Sensitivity_analysis.get_sensitivity import Sensitivity


@pytest.fixture
def experiment(model, scaler, data):
    """Fixture to set up an instance of Sensitivity for testing."""
    return Sensitivity(model, scaler, data)

def test_data_shape(experiment):
    """Test that the derivatives are the ones expected."""
    model_path = "Sensitivity_analysis/regression_model.h5"
    scaler_path = "Sensitivity_analysis/regression_scaler.gz"
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
