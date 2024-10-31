""" 
Author: Gregoire Caron 
Date Created: 2024-10-24
Last Modified: 2024-10-31
Module to build a feed-forward neural network.
"""

from tensorflow.keras import Input, layers, models
from tensorflow.keras.optimizers import Adam


def build_model(input_shape, output_shape):
    """Build a feed-forward model."""
    inputs = Input(shape=(input_shape,))
    # Use "tanh" instead of "ReLU" to avoid getting discrete gradient values
    x = layers.Dense(10, activation='tanh')(inputs)
    x = layers.Dense(10, activation='tanh')(x)
    # Use "linear" to predict negative and positive values
    outputs = layers.Dense(output_shape, activation='linear')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    opt = Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model

if __name__ == "__main__":
    build_model(input_shape=1, output_shape=1)
