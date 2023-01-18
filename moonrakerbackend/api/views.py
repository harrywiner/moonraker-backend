from rest_framework.response import Response
from rest_framework.decorators import api_view
from api.src.format import deserialize_input, map_aggregates_to_samples, is_valid_input
from api.src.ae_lib import get_sample_loss, find_zscore_anomalies
from api.src.cap_lib import get_capacities, load_battery_files, extract_battery_features, get_predicted_capacities
from api.src.rul_lib import get_predicted_capacity_feature, get_prediction_at_single_cycle

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import math

# raw = load_battery_files()
# battery_features = extract_battery_features(raw)
# capacities = [get_capacities(bat) for bat in raw]

THRESHOLD = 0.12

@api_view(['GET'])
def getData(request):
    if 'name' not in request.query_params.keys():
        person = {'name':"Joyce Lee", 'age': 22}
    else:
        name = request.query_params['name']
        person = {'name':name, 'age': 69}

    return Response(person)

@api_view(['POST'])
def findAnomalies(request):
    payload = request.data
    
    # Clean data
    clean_payload = list(filter(is_valid_input, payload))
    
    # deserialize data
    devices = [deserialize_input(obj) for obj in clean_payload]
    
    # format and reshape data
    input_data = map_aggregates_to_samples(devices)
    
    # load model
    autoencoder = load_model('moonrakerbackend/api/models/LSTM-IOT.h5')
    
    # get sample loss
    loss = get_sample_loss(input_data, autoencoder)
    
    # find anomalies 
    is_anomaly_list = find_zscore_anomalies(input_data, loss, THRESHOLD)
    
    # filter original data
    anomalies = []
    normal = []
    for (is_anomaly, obj) in zip(is_anomaly_list, clean_payload):
        if is_anomaly:
            anomalies.append(obj)
        else:
            normal.append(obj)
    res_dict = {
        "anomalies": anomalies,
        "functioning": normal
    }
    return Response(res_dict)

@api_view(['GET'])
def rulPredictor(request):
    #get parameter for t for cycles
    pass

@api_view(['POST'])
def findRUL(request):
    if 't' not in request.query_params.keys():
        cycle = 90
        
    else:
        t = request.query_params['t']
        cycle = int(t)
    
    # load model
    # cap_lstm = load_model('moonrakerbackend/api/models/CAP_DNN_test5.h5')
    
    # get predict set
    # predict_set = get_predicted_capacities(cap_lstm, battery_features, capacities)

    deserialised_data = pd.read_csv('./data/predicted_capacities.csv').to_numpy()
    deserialised_data = np.float32(deserialised_data)

    cleaned_bat18 = [x for x in deserialised_data[3] if (math.isnan(x) != True)]

    predict_set = []
    for i in deserialised_data[:3]:
        predict_set.append(i)
    predict_set.append(cleaned_bat18)

    # get predicted forward and backward
    backward, forward = get_predicted_capacity_feature(predict_set, 20, 20)

    # load model 
    rul_lstm = load_model('moonrakerbackend/api/models/RUL_LSTM_test10.h5')
    
    # find prediction
    x_indices, y_actual_all, y_predicted_all, eol_i_actual_all, eol_i_predict_all, name_all, eol_actual_all = get_prediction_at_single_cycle(rul_lstm, backward, forward, cycle = cycle)

    res_dict = {
        "predicted": y_predicted_all,
        "actual": y_actual_all,
        "x_indices" : x_indices,
        "eol_i_predict" : eol_i_predict_all,
        "eol_i_actual" : eol_i_actual_all,
        "batteries" : name_all, 
        "eol_value" : eol_actual_all
    }
    return Response(res_dict)