from api.types.data_types import AggregateDevice
from typing import List
import numpy as np

def deserialize_input(obj: dict) -> AggregateDevice:
    return AggregateDevice(
            deviceID=obj["deviceID"], 
            LatencyAvg=obj["LatencyAvg"], 
            FailureRate=obj["FailureRate"])

def map_aggregates_to_samples(devices: List[AggregateDevice]) -> List[List[List[float]]]:
    return np.array([[[device.LatencyAvg, device.FailureRate]] for device in devices])

def is_valid_input(obj: dict) -> bool:
    return not (obj["deviceID"] == None or obj["FailureRate"] == None or obj["LatencyAvg"] == None)