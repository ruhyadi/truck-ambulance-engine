"""Classifier (ambulance classifier) schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Optional

from pydantic import BaseModel, Field


class ClsResultSchema(BaseModel):
    """Classification results schema."""

    categories: List[Optional[str]] = Field([], example=["ambulance", None])
    scores: List[float] = Field([], example=[0.9, 0.8])
