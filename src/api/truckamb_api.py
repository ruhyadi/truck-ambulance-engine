"""Truck/Ambulance detection API."""

import rootutils

ROOT = rootutils.autosetup()

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from omegaconf import DictConfig
from PIL import Image

from src.schema.api_schema import TruckAmbRequestSchema, TruckAmbVidRequestSchema
from src.schema.yolo_schema import YoloResultSchema
from src.utils.logger import get_logger
from src.utils.plotter import PlotUtils

log = get_logger()


class TruckAmbApi:
    """Truck/Ambulance detection API."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize the API."""
        self.cfg = cfg
        self.app = FastAPI()
        self.router = APIRouter()

        # plot utils
        self.plotter = PlotUtils()

        self.tmp_path = Path("tmp/temp")
        self.tmp_path.mkdir(parents=True, exist_ok=True)

        self.filtered_cats = [
            "2",  # car
            "7",  # truck
        ]

        self.setup_engine()
        self.setup()

    def setup_engine(self) -> None:
        """Setup the engine."""
        log.log(21, f"Setup Truck/Ambulance detection API engine")
        yolo_engine_path: str = self.cfg.engine.truckamb.det_engine_path
        cls_engine_path: str = self.cfg.engine.truckamb.cls_engine_path

        # setup engine
        if yolo_engine_path.endswith(".plan") and cls_engine_path.endswith(".plan"):
            from src.engine.truckamb_trt_engine import TruckAmbTrtEngine

            self.engine = TruckAmbTrtEngine(**self.cfg.engine.truckamb)
        elif yolo_engine_path.endswith(".onnx") and cls_engine_path.endswith(".onnx"):
            from src.engine.truckamb_onnx_engine import TruckAmbOnnxEngine

            self.engine = TruckAmbOnnxEngine(**self.cfg.engine.truckamb)
        else:
            raise ValueError(
                f"Unsupported engine file format: {yolo_engine_path} or {cls_engine_path}"
            )

        # check categories. if null use default
        if self.engine.det_categories is None:
            self.engine.det_categories = [str(i) for i in range(80)]  # coco
        if self.engine.cls_categories is None:
            self.engine.cls_categories = [f"cls_{i}" for i in range(1000)]  # imagenet
        self.engine.setup()

    def setup(self) -> None:
        """Setup the API."""

        @self.router.post(
            "/api/v1/engine/truckamb/snapshot",
            tags=["engine"],
            summary="Truck/Ambulance detection",
        )
        async def detect(form: TruckAmbRequestSchema = Depends()):
            """Detect truck/ambulance in an image."""
            img = await self.preprocess_img_bytes(await form.image.read())
            self.plotter.setup(img)

            det_result = self.engine.predict(img)

            # plot predictions
            frame = self.plotter.draw_boxes(
                img, det_result.boxes, det_result.categories
            )

            # convert to buffer with PIL
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            buffer.seek(0)

            return StreamingResponse(buffer, media_type="image/jpeg")

        @self.router.post(
            "/api/v1/engine/truckamb/video",
            tags=["engine"],
            summary="Truck/Ambulance detection",
        )
        async def detect_video(form: TruckAmbVidRequestSchema = Depends()):
            """Detect truck/ambulance in a video."""
            vid = await form.video.read()
            vid_file = self.tmp_path / "temp.mp4"
            out_file = self.tmp_path / "output.mp4"
            with open(str(vid_file), "wb") as f:
                f.write(vid)

            # read video
            cap = cv2.VideoCapture(str(vid_file))
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            cap_fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(
                str(out_file),
                cv2.VideoWriter_fourcc(*"mp4v"),
                cap_fps,
                (frame_width, frame_height),
            )

            while cap.isOpened():
                ret, frame = cap.read()
                self.plotter.setup(frame)
                if not ret:
                    break
                det_result = self.engine.predict(frame)
                frame = self.plotter.draw_boxes(
                    frame, det_result.boxes, det_result.categories
                )
                out.write(frame)

            cap.release()
            out.release()

            return FileResponse(str(out_file))

    async def preprocess_img_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Preprocess image bytes."""
        img = Image.open(BytesIO(img_bytes))
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # if PNG, convert to RGB
        if img.shape[-1] == 4:
            img = img[..., :3]

        return img
