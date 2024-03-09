"""Plotting utils."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Optional, Tuple

import cv2
import numpy as np


class PlotUtils:
    """Plotting utils."""

    def __init__(self, line_width: Optional[int] = None) -> None:
        """Initialize."""
        self.lw = line_width
        self.colors = Colors()

    def setup(self, frame: np.ndarray) -> None:
        """Setup frame to get line width."""
        if self.lw is None:
            self.lw = max(round(sum(frame.shape) / 2 * 0.003), 2)

    def draw_boxes(
        self,
        frame: np.ndarray,
        boxes: List[List[int]],
        labels: List[str],
        colors: Optional[List[tuple]] = None,
    ) -> np.ndarray:
        """Draw bounding boxes on frame."""
        colors = (
            {label: self.colors(idx, bgr=True) for idx, label in enumerate(set(labels))}
            if colors is None
            else colors
        )

        for box, label in zip(boxes, labels):
            frame = self.draw_box(
                frame=frame,
                box=box,
                color=colors[label],
                label=label,
            )

        return frame

    def draw_box(
        self,
        frame: np.ndarray,
        box: List[int],
        color: Tuple[int, int, int],
        label: Optional[str] = None,
        txt_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Draw bounding box on frame."""
        p1, p2 = (box[0], box[1]), (box[2], box[3])

        cv2.rectangle(frame, p1, p2, color, self.lw, cv2.LINE_AA)
        if label:
            frame = self.draw_text(
                frame=frame,
                text=label,
                pos=p1,
                color=color,
            )

        return frame

    def draw_text(
        self,
        frame: np.ndarray,
        text: str,
        pos: Tuple[int, int],
        color: Tuple[int, int, int],
        txt_color: Tuple[int, int, int] = (255, 255, 255),
        box_style: bool = True,
    ) -> np.ndarray:
        """Draw text on frame."""
        tf = max(self.lw - 1, 1)
        if box_style:
            w, h = cv2.getTextSize(text, 0, fontScale=self.lw / 3, thickness=tf)[0]
            outside = pos[1] - h >= 3
            p2 = pos[0] + w, pos[1] - h - 3 if outside else pos[1] + h + 3
            cv2.rectangle(frame, pos, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            frame,
            text,
            pos,
            0,
            self.lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

        return frame


class Colors:
    """Colors pallete."""

    def __init__(self) -> None:
        """Initialize colors as hex."""
        hexs = (
            "344593",  # blue
            "FF3838",  # red
            "FF9D97",  # pink
            "FF701F",  # orange
            "FFB21D",  # yellow
            "CFD231",  # lime
            "48F90A",  # green
            "92CC17",  # green
            "3DDB86",  # green
            "1A9334",  # green
            "00D4BB",  # cyan
            "2C99A8",  # cyan
            "00C2FF",  # blue
            "6473FF",  # blue
            "0018EC",  # blue
            "8438FF",  # purple
            "520085",  # purple
            "CB38FF",  # purple
            "FF95C8",  # pink
            "FF37C7",  # pink
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, idx: int, bgr: bool = False) -> Tuple[int, int, int]:
        """Get color."""
        color = self.palette[idx % self.n]
        return color[::-1] if bgr else color

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
