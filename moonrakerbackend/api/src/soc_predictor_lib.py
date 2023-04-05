import datetime
import numpy as np
from api.src.rul_lib import find_eol
from sklearn.linear_model import LinearRegression
import csv

from typing import List

from ..types.data_types import SOC_Prediction

def read_data(file_name):
    delimiter = ','
    with open(file_name, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        all_data = []
        bat = []
        for row in reader:
            if row != []:
                cycle = []
                for i in row:
                    timesteps = np.fromstring(i.strip('[]'), sep=' ')
                    cycle.append(timesteps)
                bat.append(cycle)
            else:
                all_data.append(bat)
                bat = []
    return all_data

def train_test_ratio(features, start_charge, ratio = .1):
    all_train_x, all_train_y, all_test_x, all_test_y  = [], [], [], []
    for bat, s_c in zip(features, start_charge):
        cycle = np.array(bat)
        start_i = find_eol(cycle[:, 1], s_c)
        bat = bat[start_i:]
        split = int(len(bat)*ratio)
        train_val = bat[:split]
        test_val = bat[split:]
        train_x = np.array(train_val)[:, 0]
        train_y = np.array(train_val)[:, 1]
        test_x = np.array(test_val)[:, 0]
        test_y = np.array(test_val)[:, 1]
        all_train_x.append(train_x)
        all_train_y.append(train_y)
        all_test_x.append(test_x)
        all_test_y.append(test_y)
    return all_train_x, all_train_y, all_test_x, all_test_y

def get_soc_prediction(train_x, train_y, time, soc, start_charge, bat_names=["BAT 05", "BAT 06", "BAT 07", "BAT18"]) -> List[SOC_Prediction]:
    predictions = []
    for xs, ys, name, t_x, t_y, s_c in zip(time, soc, bat_names, train_x, train_y, start_charge):
        reg = LinearRegression()
        reg.fit(t_x, t_y)
        predict_y = reg.predict(xs)
        predict_y = [i if i > 0 else 0 for i in predict_y[:, 0]]

        ys = [i if i > 0 else 0 for i in ys[:, 0]]
        
        eol_actual = 0
        
        x_predict = find_eol(predict_y, eol_actual)
        eol_i_predict = xs[x_predict]

        x_actual = find_eol(ys, eol_actual)
        eol_i_actual = xs[x_actual]

        x_indices = np.concatenate((t_x, xs))

        predictions.append(
            SOC_Prediction(
                y_past=(t_y.flatten()).tolist(),
                y_predicted=predict_y,
                y_actual=ys,
                device_name=name, 
                x_units="Time(s)",
                y_units="Charge(%)",
                x_indices=(x_indices.flatten()).tolist(),
                time_to_empty_actual=int(eol_i_actual*1000), # milliseconds
                time_to_empty_predicted=int(eol_i_predict*1000),
                start_of_cycle=s_c)
            )
    return predictions
    

        
