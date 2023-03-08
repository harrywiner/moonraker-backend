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
    # return -1 if in the future
    return -1 

def calculate_eol(predict_y, actual_y, actual_x):
    rated_cap = actual_x[0]
    eol_actual = rated_cap *.8

    eol_i_actual = find_eol(actual_y, eol_actual)
    eol_i_predict = find_eol(predict_y, eol_actual)
    
    return eol_i_predict, eol_i_actual, eol_actual

def get_prediction_at_single_cycle(model, xs, ys, bat_names=["BAT 05", "BAT 06", "BAT 07", "BAT 18"], cycle = 0, lookback = 20, forward = 20):
    y_actual_all = []
    y_predicted_all = [] 
    past_all = []
    eol_i_actual_all = []
    eol_i_predict_all = []
    eol_actual_all = []
    name_all = []
    eol_bool_all = []
    for x, y, name in zip(xs, ys, bat_names):
        backward_indices = [i for i in range(lookback+cycle)]
        forward_indices = [i for i in range(lookback+cycle, lookback+forward+cycle)]
        predict = np.array(model.predict(x))

        x_indices = np.concatenate((backward_indices, forward_indices))
        y_actual = y[cycle]
        y_predicted = predict[cycle]

        full_history = []
        for i in x:
            full_history.append(i[0])
        past = full_history[0:cycle+lookback]

        actual_full = np.concatenate((past, y_actual))
        predict_full = np.concatenate((past, y_predicted))

        eol_i_actual, eol_i_predict, eol_actual = calculate_eol(predict_full, actual_full, past)
        eol_bool = is_eol(past, eol_actual, y_actual)

        y_actual_all.append(y_actual)
        y_predicted_all.append(y_predicted) 
        past_all.append(past)
        eol_i_actual_all.append(eol_i_actual)
        eol_i_predict_all.append(eol_i_predict)
        name_all.append(name)
        eol_actual_all.append(eol_actual)
        eol_bool_all.append(eol_bool)

    return x_indices, y_actual_all, y_predicted_all, past_all, eol_i_actual_all, eol_i_predict_all, name_all, eol_actual_all, eol_bool_all

def is_eol(past, eol_actual, actual_y):
    # if the past contains eol_actual
    if (past[-1] < eol_actual or (past[-1] > eol_actual and eol_actual > actual_y[0])):
        return True
    return False

