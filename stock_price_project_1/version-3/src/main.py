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

final_w, final_b, J_history, p_history, iteration_history = mt.gradient_descent(X_train, y_train, w_initial, 0,0.01, 2000)

mt.plot_cost_over_iterations(J_history, iteration_history)


m = X_test.shape[0]
predictions = np.zeros(m)

for i in range(m):
    # Compute the logit z
    z = np.dot(X_test[i], final_w) + final_b
    # Apply the sigmoid
    p = 1.0 / (1.0 + np.exp(-z))
    # Threshold at 0.5
    if p >= 0.5:
        predictions[i] = 1
    else:
        predictions[i] = 0

accuracy = np.mean(predictions == y_test)
print("Test Accuracy:", accuracy)

correct = np.sum(predictions == y_test)
incorrect = np.sum(predictions != y_test)

labels = ["Correct", "Incorrect"]
counts = [correct, incorrect]

plt.bar(labels, counts, color=["green", "red"])
plt.ylabel("Number of Samples")
plt.title("Logistic Regression Predictions on Test Set")
plt.show()

for i in range(10):
    print(f"Row {i}, predicted={predictions[i]}, actual={y_test[i]}")

print("Final w:", final_w)
print("Final b:", final_b)

up_fraction = np.mean(y_train)
print("Fraction of 'up' days in training:", up_fraction)
