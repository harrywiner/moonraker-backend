import numpy as np
import math
# import scipy
# from sklearn.cluster import KMeans
# import pandas as pd


# def load_battery_files(datadir="./data/"):
#     """Load the data from batteries 05, 06, 07, and 18

#     Args:
#         datadir (str, optional): Path to Directory with `5. Battery Data Set`. Defaults to "../../data/". Default to be used for Jupyter notebooks in models/RUL

#     Returns:
#         List[BatteryObject]: returns set of battery files ready for input into the `get_{variable}` methods in lib.py. Ex capacities = get_capacities(batteries[0])
#     """
#     return [scipy.io.loadmat(f'{datadir}5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0005.mat')["B0005"],
#             scipy.io.loadmat(f'{datadir}5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0006.mat')["B0006"],
#             scipy.io.loadmat(f'{datadir}5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0007.mat')["B0007"],
#             scipy.io.loadmat(f'{datadir}5. Battery Data Set/1. BatteryAgingARC-FY08Q4/B0018.mat')["B0018"]]

# def get_derivatives(x, y):
#     assert len(np.shape(y)) == 1
#     out = []
#     for i in range(len(x) - 1):
#         out.append((y[i+1] - y[i])/(x[i+1] - x[i]))
#     return out

# def get_max_initial_slope(time,variable):
#     """Get Max initial slope of function f(time) = variable.
#     To be used with x = time and y = voltage for single cycle

#     Args:
#         x (List): Shape (readings,)
#         y (List): Shape (readings,)

#     Returns:
#         float: maximum slope of y
#     """

#     assert len(np.shape(time)) == 1 and len(np.shape(variable)) == 1

#     dy_dx = get_derivatives(time,variable)

#     i = 0
#     d_max = dy_dx[i]
#     while i < len(dy_dx) - 1 and (i < 20 or abs(dy_dx[i+1]) < abs(dy_dx[i])):
#         if abs(d_max) < abs(dy_dx[i]):
#             d_max = dy_dx[i]
#         i += 1
#     return d_max

# def get_all_cycles_max_initial_slope(times,voltages):
#     """Get all ax initial slopes of cycles for function f(time) = variable.
#     To be used with x = time and y = voltage for single cycle

#     Args:
#         times (List): Shape (cycles,readings)
#         voltages (List): Shape (cycles,readings)

#     Returns:
#         List[float]: maximum slope of y for each cycle
#     """
#     return [get_max_initial_slope(t, v) for t, v in zip(times, voltages)]

# ### 5. Length of Voltage range above 3V

# def get_range_above(time, variable, threshold=3):
#     """Get Length of Domain where variable > threshold

#     Args:
#         time (List): Shape (readings,)
#         variable (List): Shape (readings,)
#         threshold (int, optional): Threshold for variable. Defaults to 3.

#     Returns:
#         float: Length of range in seconds
#     """
#     time_delta = get_time_delta(time)
#     return sum([t if v > threshold else 0 for t,v in zip(time_delta, variable)])

# def get_all_cycles_range_above(times, voltages, threshold=3):
#     """Get Length of Domain where variable > threshold for each cycle

#     Args:
#         times (List): Shape (cycles,readings)
#         voltages (List): Shape (cycles,readings)
#         threshold (int, optional): Threshold for variable. Defaults to 3.

#     Returns:
#         List[float]: Length of range in seconds for each cycle, shape (cycles,)
#     """
#     return [get_range_above(t, v) for t, v in zip(times, voltages)]

# ### 7. Discharging Time
# def get_actives(variable):
#     # warnings.filterwarnings("ignore")
#     model = KMeans(2, n_init = 10).fit(variable.reshape(len(variable),1))
    
#     if abs(model.cluster_centers_[0][0]) > abs(model.cluster_centers_[1][0]):
#         return [True if e == 0 else False for e in model.labels_]
#     else:
#         return [True if e == 1 else False for e in model.labels_]

# def get_discharge_time(time, current):
#     """Get Length of Domain where battery is actively discharging

#     Args:
#         time (List): Shape (readings,)
#         current (List): Shape (readings,)

#     Returns:
#         float: Length of active discharge Domain in seconds
#     """
#     is_active = get_actives(current)
#     delta = get_time_delta(time)
    
#     return sum([t if active else 0 for t, active in zip(delta, is_active)])


# def get_all_cycles_discharge_time(times, currents):
#     """Get Length of Domain where battery is actively discharging for each cycle

#     Args:
#         times (List): Shape (cycles,readings)
#         currents (List): Shape (cycles,readings)

#     Returns:
#         List[float]: Length of active discharge Domain in seconds for each cycle
#     """
#     return [get_discharge_time(t, c) for t, c in zip(times, currents)]

# def get_time_delta(time):
#     return [0, *[time[i+1] - time[i] for i in range(len(time) - 1)]]

# def get_voltages(raw):
#     """Get all voltages for all cycles

#     Args:
#         raw (List): raw Bat file after subscripting, ex read_file['B0005']

#     Returns:
#         List: Shape (cycles, [variable cycle length])
#     """
#     discharges = list(filter(lambda e: e[0][0] == "discharge",raw[0][0][0][0]))
#     return [e[3][0][0][0][0] for e in discharges]

# def get_current(raw):
#     """Get all currents for all cycles

#     Args:
#         raw (List): raw Bat file after subscripting, ex read_file['B0005']

#     Returns:
#         List: Shape (cycles, [variable cycle length])
#     """
#     discharges = list(filter(lambda e: e[0][0] == "discharge",raw[0][0][0][0]))
#     return [e[3][0][0][1][0] for e in discharges]

# def get_time(raw):
#     """Get all time stamps for all readings in cycles

#     Args:
#         raw (List): raw Bat file after subscripting, ex read_file['B0005']

#     Returns:
#         List: Shape (cycles, [variable cycle length])
#     """
#     discharges = list(filter(lambda e: e[0][0] == "discharge",raw[0][0][0][0]))
#     return [e[3][0][0][5][0] for e in discharges]

# def extract_battery_features(batteries):
#     output = []
#     for bat in batteries:
#         # Get Raw Variables
#         times = get_time(bat)
#         voltages = get_voltages(bat)
#         cur = get_current(bat)
        
#         # Feature Extraction
#         max_initials = get_all_cycles_max_initial_slope(times, voltages)
#         discharge_times = get_all_cycles_discharge_time(times, cur)
#         ranges_above = get_all_cycles_range_above(times, voltages)
        
#         # Assert similar Shapes
#         assert np.shape(max_initials) == np.shape(ranges_above)
#         assert np.shape(discharge_times) == np.shape(ranges_above)
        
#         # Stack Features
#         features = np.stack((max_initials, ranges_above, discharge_times), axis = 1)
#         features = features.reshape(features.shape[0], 1, features.shape[1])
        
#         output.append(features)
        
#     return output

# def get_capacities(raw):
#     """Get all capacities for all cycles

#     Args:
#         raw (List): raw Bat file after subscripting, ex read_file['B0005']

#     Returns:
#         List: Shape (cycles, 1)
#     """
#     discharges = list(filter(lambda e: e[0][0] == "discharge",raw[0][0][0][0]))
#     return [e[3][0][0][6][0] for e in discharges]

# def get_predicted_capacities(model, xs, ys, bat_names=["BAT 05", "BAT 06", "BAT 07", "BAT 18"]):
#     predict_full_set = []
#     for x, y, name in zip(xs, ys, bat_names):
#         predict = np.array(model.predict(x)).flatten()
#         predict_full_set.append(predict)

#     return predict_full_set

def clean_data(deserialized_data):
    
    deserialized_data = np.float32(deserialized_data)
    cleaned_bat18 = [x for x in deserialized_data[3] if (math.isnan(x) != True)]

    predict_set = []
    for i in deserialized_data[:3]:
        predict_set.append(i)
    predict_set.append(cleaned_bat18)

    return predict_set



