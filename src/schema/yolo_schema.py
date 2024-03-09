"""YOLO engine schema."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

from pydantic import BaseModel, Field, model_validator


class YoloResultSchema(BaseModel):
    """YOLO engine detection result schema."""

    boxes: List[List[int]] = Field([], example=[[0, 0, 100, 100], [50, 50, 150, 150]])
    scores: List[float] = Field([], example=[0.9, 0.8])
    categories: List[str] = Field([], example=["person", "car"])

    @model_validator(mode="after")
    def validator(self) -> "YoloResultSchema":
        """Clip boxes to image size and round scores."""
        self.scores = [round(x, 2) for x in self.scores]
        self.boxes = [[max(x, 1) for x in box] for box in self.boxes]

        return self
