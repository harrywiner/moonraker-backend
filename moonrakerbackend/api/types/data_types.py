from pydantic import BaseModel

class AggregateDevice(BaseModel):
    deviceID: str
    LatencyAvg: float
    FailureRate: float