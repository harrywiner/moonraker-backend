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

def train_test_ratio(features, ratio = .1):
    all_train_x, all_train_y, all_test_x, all_test_y  = [], [], [], []
    for bat in features:
        bat_train_x = []
        bat_train_y = []
        bat_test_x = []
        bat_test_y = []
        for cycle in bat:
            split = int(len(cycle)*ratio)
            train_val = cycle[:split]
            test_val = cycle[split:]
            train_x = np.array(train_val)[:, 0]
            train_y = np.array(train_val)[:, 1]
            test_x = np.array(test_val)[:, 0]
            test_y = np.array(test_val)[:, 1]
            bat_train_x.append(train_x)
            bat_train_y.append(train_y)
            bat_test_x.append(test_x)
            bat_test_y.append(test_y)
        all_train_x.append(bat_train_x)
        all_train_y.append(bat_train_y)
        all_test_x.append(bat_test_x)
        all_test_y.append(bat_test_y)
    return all_train_x, all_train_y, all_test_x, all_test_y

def get_soc_prediction(train_x, train_y, time, soc, bat_names=["BAT 05", "BAT 06", "BAT 07", "BAT18"], cycles= [0,0,0,0]) -> List[SOC_Prediction]:
    predictions = []
    for xs, ys, name, t_x, t_y, c in zip(time, soc, bat_names, train_x, train_y, cycles):
        x = xs[c].reshape((-1, 1))
        reg = LinearRegression()
        t = t_x[c].reshape((-1, 1))
        s = t_y[c]
        reg.fit(t, s)
        predict_y = reg.predict(x)
        predict_y = [i if i > 0 else 0 for i in predict_y]
        
        eol_actual = 0
        
        x_predict = find_eol(predict_y, eol_actual)
        eol_i_predict = xs[c][x_predict]

        x_actual = find_eol(ys[c], eol_actual)
        eol_i_actual = xs[c][x_actual]

        x_indices = np.concatenate((t_x[c], xs[c]))

        predictions.append(
            SOC_Prediction(
                y_past=t_y[c].tolist(),
                y_predicted=predict_y,
                y_actual=ys[c].tolist(),
                device_name=name, 
                x_units="Time(s)",
                y_units="Charge(%)",
                x_indices=x_indices.tolist(),
                time_to_empty_actual=int(eol_i_actual*1000), # milliseconds
                time_to_empty_predicted=int(eol_i_predict*1000),
                cycle = c)
            )
    return predictions
    

        
