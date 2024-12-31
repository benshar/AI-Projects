import matplotlib.pyplot as plt
import math
import numpy as np
import copy

def cost_function(X, y, w, b):

    m = X.shape[0]
    total_cost = 0
    for i in range(m):
        f_wb = np.dot(w,X[i]) + b
        cost = (f_wb - y[i]) ** 2
        total_cost += cost
    total_cost = total_cost / (2*m)
    return total_cost

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

def plot_cost_over_iterations(J_history, iteration_history):
    plt.plot(iteration_history, J_history, label='Cost over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost over Iterations')
    plt.show()
