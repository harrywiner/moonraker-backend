from rest_framework.response import Response
from rest_framework.decorators import api_view
from api.src.format import deserialize_input, map_aggregates_to_samples, is_valid_input
from api.src.ae_lib import get_sample_loss, find_zscore_anomalies
from api.src.cap_lib import clean_data
from api.src.rul_lib import get_predicted_capacity_feature, get_prediction_at_single_cycle
from api.types.data_types import SOC_Prediction
from api.src.soc_predictor_lib import read_data, train_test_ratio, get_soc_prediction

from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import ast

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
        cycle = [90]*4
    else:
        t = request.query_params['t']
        if t.isnumeric() == True:
            cycle = [int(t)]*4
        else:
            cycle = ast.literal_eval(t)

    # load model
    # cap_lstm = load_model('moonrakerbackend/api/models/CAP_DNN_test5.h5')
    
    # get predict set
    # predict_set = get_predicted_capacities(cap_lstm, battery_features, capacities)

    # deserialize data
    deserialized_data = pd.read_csv('./data/predicted_capacities.csv').to_numpy()

    # clean data
    predict_set = clean_data(deserialized_data)

    # get predicted forward and backward
    backward, forward = get_predicted_capacity_feature(predict_set, 20, 20)

    # load model 
    rul_lstm = load_model('moonrakerbackend/api/models/RUL_LSTM_test10.h5')
    
    # find prediction
    predictions = get_prediction_at_single_cycle(rul_lstm, backward, forward, cycles = cycle)

    return Response([p.dict() for p in predictions])

@api_view(['POST'])
def predictSOC(request) -> SOC_Prediction:
    if 's' not in request.query_params.keys():
        start_charge = [80]*4
    else:
        s = request.query_params['s']
        if s.isnumeric() == True:
            start_charge = [int(s)]*4
        else:
            start_charge = ast.literal_eval(s)

    calculated_cap = [i for i in read_data('./data/calculated_cap.csv')]

    train_x, train_y, test_x, test_y = train_test_ratio(calculated_cap, start_charge = start_charge)

    predictions = get_soc_prediction(train_x, train_y, test_x, test_y, start_charge = start_charge)

    return Response([p.dict() for p in predictions])