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
            outputs = [self.model(x_tensor)[:,i] for i in range(2)]
        # Divide by scaler scale to get back true gradient
        # self.dy_dx shape = (len(self.data), nb of inputs)
        self.dy_dx = [t.gradient(outputs[i], x_tensor)/(self.scaler.scale_) for i in range(2)]

    def plot_gradient(self, i, j):
        """Plot the gradient component dyj_dxi."""
        plt.scatter(self.data[:,i], self.dy_dx[j][:,i])
        plt.grid()
        plt.tight_layout()
        plt.show()

    def get_measures(self, i, j):
        """Compute additional sensitivity measures."""
        mean_sens = np.mean(self.dy_dx[j][:,i])
        sens_std = np.std(self.dy_dx[j][:,i])
        mean_sqd_sens = np.sqrt(np.mean(self.dy_dx[j][:,i]))
        return mean_sens, sens_std, mean_sqd_sens
