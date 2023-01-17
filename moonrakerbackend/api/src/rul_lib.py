import numpy as np
import pandas as pd

def get_predicted_capacity_feature(predict, backward_window, forward_window):
    backward = [look_back_reversed(backward_window, c, forward_window) for c in predict]
    forward = [look_forward(forward_window, c, backward_window) for c in predict]
    
    return backward, forward

def look_forward(n, data, truncate_past=0):
    df = pd.DataFrame(data)
    future = pd.DataFrame()
    for i in range(1, n+1):
        future[i] = df[0].shift(-(i))
    return future[truncate_past-1:-n].to_numpy()

def look_back_reversed(n, data, truncate_future):
    df = pd.DataFrame(data)
    past = pd.DataFrame()

    for i in range(n-1, 0, -1):
        past[-i] = df[0].shift(i)
    past[0] = df[0]

    return past[n-1:-truncate_future].to_numpy()

def find_eol(vector, eol_actual):
    # Find first cycle that passes EOL threshold
    for i in range(len(vector) - 1):
        if (vector[i] > eol_actual and vector[i+1] < eol_actual):
            # print("First capacity that passes threshold:", vector[i])
            return i+1
    return -1

def calculate_eol(predict_y, actual_y, actual_x):
    rated_cap = actual_x[0, 0]
    eol_actual = rated_cap *.8

    eol_i_actual = find_eol(actual_y, eol_actual)
    eol_i_predict = find_eol(predict_y, eol_actual)
    
    return eol_i_predict, eol_i_actual, eol_actual

def get_prediction_at_single_cycle(model, xs, ys, bat_names=["BAT 05", "BAT 06", "BAT 07", "BAT 18"], cycle=0, lookback = 20, forward = 20):
    for x, y, name in zip(xs, ys, bat_names):
        backward_indices = [i for i in range(cycle, lookback+cycle)]
        forward_indices = [i for i in range(lookback+cycle, lookback+forward+cycle)]
        predict = np.array(model.predict(x))

        rated_cap = x[0, 0]
        eol_actual = rated_cap *.8

        actual_full = np.concatenate((x[cycle], y[cycle]))
        predict_full = np.concatenate((x[cycle], predict[cycle]))

        eol_i_predict, eol_i_actual, eol_actual= calculate_eol(predict_full, actual_full, x)
    
    return x[cycle], y[cycle], predict[cycle], eol_i_predict, eol_i_actual, backward_indices, forward_indices
