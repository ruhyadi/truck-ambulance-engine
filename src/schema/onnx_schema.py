"""ONNX engine schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import numpy as np
from pydantic import BaseModel, Field, model_validator


class OnnxMetadataSchema(BaseModel):
    """ONNX metadata schema."""

    input_name: str = Field(..., example="images")
    input_shape: List = Field(..., example=[1, 3, 224, 224])
    output_name: str = Field(..., example="output")
    output_shape: List = Field(..., example=[1, 8400, 85])

    @model_validator(mode="after")
    def check_shape(self) -> "OnnxMetadataSchema":
        """If dynamic shape, set to -1."""
        self.input_shape = [-1 if isinstance(i, str) else i for i in self.input_shape]
        self.output_shape = [-1 if isinstance(i, str) else i for i in self.output_shape]

        return self


class OnnxRunEnd2EndOutputSchema(BaseModel):
    """
    ONNX run end-to-end output schema.
    Supported models: yolov6, yolov8
    """

    num_det: int = Field(..., example=20)
    boxes: np.ndarray = Field(..., example=[0, 0, 100, 100, 50, 50, 150, 150])
    scores: np.ndarray = Field(..., example=[0.9, 0.8])
    classes: np.ndarray = Field(..., example=[0, 1])

    class Config:
        arbitrary_types_allowed = True
