import matplotlib.pyplot as plt
import math

def standard_deviation(data_frame):
    average = data_frame.iloc[:, 0].mean()
    sd_sum = 0
    for i in data_frame.iloc[:, 0]:
        subtraction = (i - average) ** 2
        sd_sum += subtraction
    multiplying = sd_sum * (1/data_frame.shape[0])
    sd = math.sqrt(multiplying)
    return sd

def feature_scaling(x_train, y_train, X_test, y_test):
    sd_x = standard_deviation(x_train)
    sd_y = standard_deviation(y_train)
    mean_x = x_train.iloc[:,0].mean()
    mean_y = y_train.iloc[:, 0].mean()

    x_train_scaled = x_train.copy()
    y_train_scaled = y_train.copy()

    x_train_scaled.drop(x_train_scaled.columns[0], axis=1, inplace=True)
    x_train_scaled.insert(0, x_train.columns[0], ((x_train.iloc[:, 0] - mean_x) / sd_x).astype(float))
    y_train_scaled.iloc[:, 0] = ((y_train.iloc[:, 0] - mean_y) / sd_y).astype(float)
    X_test_scaled = ((X_test.iloc[:, 0] - mean_x) / sd_x).to_frame()
    y_test_scaled = ((y_test.iloc[:, 0] - mean_y) / sd_y).to_frame()
    return x_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled

def cost_function(x_train, y_train, w, b):
    cost = 0
    for i in range(x_train.shape[0]):
        f_wb = w*x_train.iloc[i,0] + b
        cost += ((f_wb - y_train.iloc[i,0])** 2)
    divided_cost = (1/2 * (1/x_train.shape[0])) * cost
    return divided_cost

def gradient_calculation(x_train, y_train, w, b):
    dj_dw = 0
    dj_db = 0
    for i in range(x_train.shape[0]):
        f_wb = w * x_train.iloc[i, 0] + b
        dj_dw += (f_wb - y_train.iloc[i,0]) * x_train.iloc[i,0]
        dj_db += (f_wb - y_train.iloc[i,0])
    divided_dj_dw = (1/x_train.shape[0]) * dj_dw
    divided_dj_db = (1 / x_train.shape[0]) * dj_db
    return divided_dj_dw, divided_dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    J_history = []
    p_history = []
    iteration_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        dj_dw, dj_db = gradient_calculation(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 10 == 0 and i < 100000:
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])
            iteration_history.append(i)
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history, iteration_history


def plot_cost_over_iterations(J_history, iteration_history):
    plt.plot(iteration_history, J_history, label='Cost over Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost over Iterations')
    plt.show()

def unscale(w, b, X_scaled, y_train):
    y_pred_scaled = w * X_scaled + b
    y_pred_unscaled = y_pred_scaled * standard_deviation(y_train) + y_train.iloc[:, 0].mean()
    return y_pred_unscaled
