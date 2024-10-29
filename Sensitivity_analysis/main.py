import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib

from Models.feed_forward import build_model
from get_sensitivity import Sensitivity


class Experiment():

    def __init__(self, N=1000, test_size=0.20):
        self.N = N
        self.test_size=test_size
        self.model = None
        self.scaler = StandardScaler()
        self.X_train, self.X_test, self.y_train, self.y_test = self.get_data()
        self.scale_data()

    def get_data(self):
        """Get train and test datasets."""
        X = np.random.rand(self.N)*2*np.pi
        X = X.reshape(-1, 1)
        y = np.sin(X)+X
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=23)
        return X_train, X_test, y_train, y_test
    
    def get_data_2(self):
        """Get train and test datasets."""
        X1 = (np.random.rand(self.N)*2*np.pi).reshape(-1, 1)
        X2 = (np.random.rand(self.N)).reshape(-1, 1)
        X = np.concatenate((X1, X2), axis=1)
        y = np.sin(X1)+X2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=23)
        return X_train, X_test, y_train, y_test

    def scale_data(self):
        """Scale inputs to have zero mean and unit std."""
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def build_and_train_model(self):
        """Build regression model."""
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        self.model = build_model(input_shape=self.X_train.shape[1], output_shape=self.y_train.shape[1])
        self.model.fit(
            self.X_train, self.y_train,
            batch_size=32, epochs=1000, validation_split=0.20,
            verbose=1, callbacks=[es])
        
    def save_model_and_scaler(self, model_path, scaler_path):
        """Save trained regression model."""
        self.model.save(model_path)
        joblib.dump(self.scaler, scaler_path)

    def plot_predictions(self):
        """Compare model predictions to true test values."""
        y_pred = self.model.predict(self.X_test)
        plt.scatter(self.y_test, y_pred)
        plt.grid()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    # Build experiment
    expe = Experiment()
    expe.build_and_train_model()
    model_path = "Sensitivity_analysis/regression_model.h5"
    scaler_path = "Sensitivity_analysis/regression_scaler.gz"
    expe.save_model_and_scaler(model_path, scaler_path)
    expe.plot_predictions()

    # Compute sensitivity values
    data = expe.scaler.inverse_transform(expe.X_test)
    sens = Sensitivity(expe.model, expe.scaler, data)
    sens.compute_gradient()
    sens.plot_gradient(0)
    sens.plot_gradient(1)
    mu, sigma, mu_sqd = sens.get_measures()
