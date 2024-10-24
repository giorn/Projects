import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
import tensorflow as tf

from Models import feed_forward


# Create data
N = 1000
X = np.random.rand(N)*2*np.pi
X = X.reshape(-1, 1)
y = np.sin(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=23)

# Scale inputs
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Build model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
model = feed_forward.build_model(input_shape=1, output_shape=1)
history = model.fit(X_train, y_train, batch_size=32, epochs=1000, validation_split=0.20, verbose=1, callbacks=[es])

# Get predictions
y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.grid()
plt.tight_layout()
plt.show()

# Compute gradient
x_tensor = tf.constant(X_test, dtype=tf.float64)
with tf.GradientTape() as t:
    t.watch(x_tensor)
    output = model(x_tensor)
# Divide by scaler scale to get back true gradient
dy_dx = t.gradient(output, x_tensor)/(scaler.scale_)

# Plot gradient
X_test = scaler.inverse_transform(X_test)
plt.scatter(X_test, np.cos(X_test))
plt.scatter(X_test, dy_dx)
plt.grid()
plt.tight_layout()
plt.show()
