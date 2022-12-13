from rest_framework.response import Response
from rest_framework.decorators import api_view
from api.src.format import deserialize_input, map_aggregates_to_samples
from api.src.ae_lib import get_sample_loss, find_zscore_anomalies

from tensorflow.keras.models import load_model

THRESHOLD = 0

@api_view(['GET'])
def getData(request):
    person = {'name':"John", 'age': 21}
    return Response(person)

@api_view(['POST'])
def findAnomalies(request):
    payload = request.data
    
    # deserialize data
    devices = [deserialize_input(obj) for obj in payload]
    
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
    for (is_anomaly, obj) in zip(is_anomaly_list, payload):
        if is_anomaly:
            anomalies.append(obj)
    
    return Response(anomalies)