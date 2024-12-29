import data_collection as dc
import model_training as mt
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

from model_training import standard_deviation

X_pre_fs = dc.X_train
y_pre_fs = dc.y_train
X_test_pre_fs = dc.X_test
y_test_pre_fs = dc.y_test

X_train, y_train, X_test, y_test  = mt.feature_scaling(X_pre_fs, y_pre_fs, X_test_pre_fs, y_test_pre_fs)

final_w, final_b, J_history, p_history, iteration_history = mt.gradient_descent(X_train, y_train, 0, 0,0.05, 200)

mt.plot_cost_over_iterations(J_history, iteration_history)

mean_x = X_pre_fs.iloc[:,0].mean()
std_x = standard_deviation(X_pre_fs)
mean_y = y_pre_fs.iloc[:,0].mean()
std_y = standard_deviation(y_pre_fs)

def predict_price_on_date(date_str):
    month, day, year = date_str.split('/')
    input_date = datetime.datetime(int(year), int(month), int(day))
    ref_date = datetime.datetime(2004, 1, 1)
    delta = input_date - ref_date
    days_since_2004 = delta.days
    x_input_scaled = (days_since_2004 - mean_x) / std_x
    y_pred_scaled = final_w * x_input_scaled + final_b
    y_pred_unscaled = (y_pred_scaled * std_y) + mean_y

    return y_pred_unscaled

def ask_for_input():
    date = input("What date would you like a prediction of? (MM/DD/YYYY format)\n")
    prediction = predict_price_on_date(date)
    print(f"The S&P 500 will be valued at {prediction} on {date}.")
    cont = input("Would you like to continue? (Yes/No)\n")
    if cont.lower() == "yes":
        ask_for_input()
    else:
        print("Thanks for playing!")

ask_for_input()
