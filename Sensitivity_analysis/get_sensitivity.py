import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


class Sensitivity():

    def __init__(self, model, scaler, data):
        self.model = model
        self.scaler = scaler
        self.data = data

    def compute_gradient(self):
        """Compute the gradient of outputs w.r.t. inputs"""
        scaled_data = self.scaler.transform(self.data)
        x_tensor = tf.constant(scaled_data, dtype=tf.float64)
        with tf.GradientTape(persistent=True) as t:
            t.watch(x_tensor)
            output = self.model(x_tensor)
        # Divide by scaler scale to get back true gradient
        # self.dy_dx shape = (len(self.data), nb of inputs)
        self.dy_dx = t.gradient(output, x_tensor)/(self.scaler.scale_)

    def plot_gradient(self, i):
        """Plot the gradient."""
        plt.scatter(self.data[:,i], self.dy_dx[:,i])
        plt.grid()
        plt.tight_layout()
        plt.show()

    def get_measures(self):
        mean_sens = np.mean(self.dy_dx)
        sens_std = np.std(self.dy_dx)
        mean_sqd_sens = np.sqrt(np.mean(self.dy_dx))
        return mean_sens, sens_std, mean_sqd_sens
