from typing import List
from pydantic import BaseModel

class AggregateDevice(BaseModel):
    deviceID: str
    LatencyAvg: float
    FailureRate: float

class Prediction(BaseModel):
    y_past: List[float]
    y_predicted: List[float]
    y_actual: List[float]
    device_name: str
    x_units: str
    y_units: str
    x_indices: List[float]


class RUL_Prediction(Prediction):
    eol_i_predict: int
    eol_i_actual: int
    eol_value: float
    is_eol: bool
    cycle: int