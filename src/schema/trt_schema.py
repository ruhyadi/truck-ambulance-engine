"""TensorRT schema."""

import rootutils

ROOT = rootutils.autosetup()

import numpy as np
from pycuda.driver import DeviceAllocation
from pydantic import BaseModel, Field


# create host memory buffer
class HostMemBufferSchema(BaseModel):
    """Host memory buffer for TensorRT inference."""

    host: np.ndarray = Field(..., description="Host memory buffer")
    device: DeviceAllocation = Field(..., description="Device memory buffer")
    binding: str = Field(..., description="Binding name")

    class Config:
        arbitrary_types_allowed = True


class End2EndResultSchema(BaseModel):
    """
    TensorRT forward end-to-end result schema.
    Supported models: yolov6, yolov8
    """

    num_det: int = Field(..., example=20)
    boxes: np.ndarray = Field(..., example=[0, 0, 100, 100, 50, 50, 150, 150])
    scores: np.ndarray = Field(..., example=[0.9, 0.8])
    categories: np.ndarray = Field(..., example=[0, 1])

    class Config:
        arbitrary_types_allowed = True
