"""Truck-ambulance onnx detection module."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List

import cv2
import numpy as np

from src.engine.amb_cls_onnx_engine import AmbulanceClsOnnxEngine
from src.engine.yolo_onnx_engine import YoloOnnxEngine
from src.schema.yolo_schema import YoloResultSchema
from src.utils.logger import get_logger

log = get_logger()


class TruckAmbOnnxEngine:
    """Truck-ambulance onnx detection module."""

    def __init__(
        self,
        det_engine_path: str,
        cls_engine_path: str,
        cls_max_batch_size: int,
        det_categories: List[str],
        cls_categories: List[str],
        det_provider: str = "cpu",
        cls_provider: str = "cpu",
        det_end2end: bool = True,
        det_arch: str = "yolov8",
        det_pretrained: bool = True,
    ) -> None:
        """
        Initialize Truck-ambulance onnx detection module.

        Args:
            det_engine_path (str): Path to detection engine.
            cls_engine_path (str): Path to classification engine.
            cls_max_batch_size (int): Maximum batch size for classification engine.
            det_categories (List[str]): List of detection categories.
            cls_categories (List[str]): List of classification categories.
            det_provider (str, optional): Inference provider for detection. Defaults to "cpu".
            cls_provider (str, optional): Inference provider for classification. Defaults to "cpu".
            det_end2end (bool, optional): Whether to use end2end model for detection. Defaults to True.
            det_arch (str, optional): Yolo architecture for detection. Defaults to "yolox".
            det_pretrained (bool, optional): Whether to use pretrained model for detection. Defaults to False.
        """
        self.det_engine_path = det_engine_path
        self.cls_engine_path = cls_engine_path
        self.cls_max_batch_size = cls_max_batch_size
        self.det_categories = det_categories
        self.cls_categories = cls_categories
        self.det_provider = det_provider
        self.cls_provider = cls_provider
        self.det_end2end = det_end2end
        self.det_arch = det_arch
        self.det_pretrained = det_pretrained

        self.filterd_cats = [
            "2",  # car
            "7",  # truck
        ]

    def setup(self) -> None:
        """Setup Truck-ambulance onnx detection module."""
        log.info(f"Setup truck-ambulance onnx engine")
        self.det_engine = YoloOnnxEngine(
            engine_path=self.det_engine_path,
            categories=self.det_categories,
            provider=self.det_provider,
            end2end=self.det_end2end,
            arch=self.det_arch,
            pretrained=self.det_pretrained,
        )
        self.det_engine.setup()

        self.cls_engine = AmbulanceClsOnnxEngine(
            engine_path=self.cls_engine_path,
            max_batch_size=self.cls_max_batch_size,
            categories=self.cls_categories,
            provider=self.cls_provider,
        )
        self.cls_engine.setup()

        log.info(f"Truck-ambulance onnx engine setup complete")

    def predict(
        self,
        img: np.ndarray,
        det_conf: float = 0.25,
        cls_conf: float = 0.25,
        det_nms: float = 0.45,
    ) -> YoloResultSchema:
        """
        Predict detection and classification on image(s).

        Args:
            imgs (List[str]): Image(s) to predict.
            det_conf (float, optional): Detection confidence threshold. Defaults to 0.25.
            cls_conf (float, optional): Classification confidence threshold. Defaults to 0.25.
            det_nms (float, optional): Detection NMS threshold. Defaults to 0.45.

        Returns:
            YoloResultSchema: Detection result with classification.
        """
        # detect objects
        det_result = self.det_engine.predict(img, conf=det_conf, nms=det_nms)[0]

        # classify detected objects
        cls_imgs = self.preprocess_cls(img, det_result)
        cls_result = self.cls_engine.predict(imgs=cls_imgs, conf=cls_conf)

        # filter by categories
        det_result_ = YoloResultSchema()
        for i, cat in enumerate(det_result.categories):
            if cat in self.filterd_cats:
                det_result_.boxes.append(det_result.boxes[i])
                det_result_.categories.append(cat)
                det_result_.scores.append(det_result.scores[i])
        det_result = det_result_

        # merge classification result with detection result
        det_result.categories = [
            cat if cat else det_result.categories[i]
            for i, cat in enumerate(cls_result.categories)
        ]

        return det_result

    def preprocess_cls(
        self, img: np.ndarray, det_result: YoloResultSchema
    ) -> List[np.ndarray]:
        """
        Preprocess image for classification.

        Args:
            img (np.ndarray): Image to preprocess.
            det_result (YoloResultSchema): Detection result.

        Returns:
            List[np.ndarray]: List of cropped images.
        """
        results: List[np.ndarray] = []
        for box in det_result.boxes:
            crop_img = img[box[1] : box[3], box[0] : box[2]]
            crop_img = cv2.resize(crop_img, self.cls_engine.img_shape)
            results.append(crop_img)

        return results
