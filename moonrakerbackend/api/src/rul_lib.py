import numpy as np
import pandas as pd

from typing import List

from ..types.data_types import RUL_Prediction

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

def get_prediction_at_single_cycle(model, xs, ys, bat_names=["BAT 05", "BAT 06", "BAT 07", "BAT 18"], cycles = [0,0,0,0], lookback = 20, forward = 20) -> List[RUL_Prediction]:

    predictions = []
    for x, y, name, c in zip(xs, ys, bat_names, cycles):

        backward_indices = [i for i in range(lookback+c)]
        forward_indices = [i for i in range(lookback+c, lookback+forward+c)]
        predict = np.array(model.predict(x))

        x_indices = np.concatenate((backward_indices, forward_indices))
        y_actual = y[c]
        y_predicted = predict[c]

        full_history = []
        for i in x:
            full_history.append(i[0])
        past = full_history[0:c+lookback]

        actual_full = np.concatenate((past, y_actual))
        predict_full = np.concatenate((past, y_predicted))

        eol_i_actual, eol_i_predict, eol_actual = calculate_eol(predict_full, actual_full, past)
        eol_bool = is_eol(past, eol_actual, y_actual)

        predictions.append(
            RUL_Prediction(
                y_past=past, 
                y_predicted=y_predicted.tolist(), 
                y_actual=y_actual.tolist(), 
                device_name=name, 
                x_units="Cycles", 
                y_units="mAh", 
                x_indices=x_indices.tolist(), 
                eol_i_predict=eol_i_predict,
                eol_i_actual=eol_i_actual, 
                eol_value=eol_actual, 
                is_eol=eol_bool, 
                cycle=c)
            )

    return predictions

def is_eol(past, eol_actual, actual_y):
    # if the past contains eol_actual
    if (past[-1] < eol_actual):
        return True
    return False

