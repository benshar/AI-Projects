# Project Explanation

The purpose of this project is to forecast the next day’s open price for the S&P 500 using multiple linear regression, incorporating various technical indicators. Single-feature ‘days_since_2004’ was too simplistic. Now, I explored how multiple features (moving averages, RSI, etc.) can yield more realistic predictions. This project has a couple components, of which are explained below.

+ **src:** The src folder consists of the source code on which the model is trained. It includes:
  + **__init__.py:** Basic empty __init__.py file.
  + **data_collection.py:** File for collecting original SP500 ticker data. I also attempted to collect VIX data, but that ended up showing up as NaN.
  + **model_training.py:** File that establishes the functions for the actual training of the model.
  + **main.py:** File that executes the training and plots comparisons.
+ **README.md:** Goes over what this project covers, how it differs from version 1, and how to run it.
+ **project_explanation.md:** The current document.
+ **requirements.txt:** List of dependencies needed to run this model.
 
## Key Functions

**Cost Function**
```python
def cost_function(X, y, w, b):

    m = X.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = np.dot(w,X[i]) + b
        cost = (f_wb - y[i]) ** 2
        total_cost += cost
    total_cost = total_cost / (2*m)
    return total_cost
```
This function calculates the total cost, or difference between the predicted values that the model gives with it current weight and bias vs the actual y value of a set. While not needed for gradient descent, it is good to graph and keep track of how the model is performing.

**Gradient Calculation Function**
```python
def gradient_calculation(X, y, w, b):

    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        for j in range(n):
            dj_dw_i = (f_wb - y[i]) * X[i, j]
            dj_dw[j] += dj_dw_i
        dj_db_i = (f_wb - y[i])
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db
```
This function finds the gradient that needs to be applied based on the derivative of the current weight and bias.

**Gradient Descent Function**
```python
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    J_history = []
    p_history = []
    iteration_history = []
    b = b_in
    w = copy.deepcopy(w_in)

    for i in range(num_iters):
        dj_dw, dj_db = gradient_calculation(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 10 == 0 and i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
            iteration_history.append(i)
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ")

    return w, b, J_history, p_history, iteration_history
```
This function actually performs gradient descent using the gradient calculation function last defined. It readjusts the weight based on the new gradient, and also tracks the cost on each 10th iteration of descent.

**Plot Cost Over Iterations Function**
```python
def plot_cost_over_iterations(J_history, iteration_history):
    plt.plot(iteration_history, J_history, label='Cost over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost over Iterations')
    plt.show()
```
This function shows if the cost is converging over a certain amount of iterations.

**Actual Y vs Predicted Y Plot (not a function)**
```python
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
```
While not a function, this plots how accurate the model's predictions on the test set are to the actual y models. It turned out to be fairly accurate, as can be seen below.

![image](https://github.com/user-attachments/assets/49ed747f-0afc-4ff1-82b2-b4551c9e3270)

