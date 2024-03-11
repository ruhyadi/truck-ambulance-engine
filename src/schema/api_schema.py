"""Truck ambulance API schema."""

import rootutils

ROOT = rootutils.autosetup()

from fastapi import File, UploadFile
from pydantic import BaseModel, Field


class TruckAmbRequestSchema(BaseModel):
    """Truck ambulance engine request."""

    image: UploadFile = File(...)
    detThreshold: float = Field(0.25, ge=0, le=1)
    clsThreshold: float = Field(0.25, ge=0, le=1)


class TruckAmbVidRequestSchema(BaseModel):
    """Truck ambulance engine request."""

    video: UploadFile = File(...)
    detThreshold: float = Field(0.25, ge=0, le=1)
    clsThreshold: float = Field(0.25, ge=0, le=1)
