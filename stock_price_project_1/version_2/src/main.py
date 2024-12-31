import data_collection as dc
import model_training as mt
import datetime
import numpy as np
import matplotlib.pyplot as plt

X_train = dc.X_train
y_train = dc.y_train
X_test = dc.X_test
y_test = dc.y_test


w_initial = np.zeros(X_train.shape[1])

final_w, final_b, J_history, p_history, iteration_history = mt.gradient_descent(X_train, y_train, w_initial, 0,0.05, 2000)

mt.plot_cost_over_iterations(J_history, iteration_history)

m = X_test.shape[0]

predictions = np.zeros(m)

for i in range(m):
    f_wb = np.dot(X_test[i],final_w) + final_b
    predictions[i] = f_wb

plt.scatter(y_test, predictions, label="Predicted")
plt.xlabel("Actual y")
plt.ylabel("Predicted y")
plt.legend()
plt.show()
