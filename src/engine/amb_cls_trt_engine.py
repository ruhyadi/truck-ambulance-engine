"""Ambulance classifier tensorrt engine."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

import cv2
import numpy as np

from src.engine.trt_engine import TrtEngine
from src.schema.cls_schema import ClsResultSchema
from src.schema.trt_schema import HostMemBufferSchema
from src.utils.logger import get_logger

log = get_logger()


class AmbulanceClsTrtEngine(TrtEngine):
    """Ambulance classifier tensorrt engine."""

    def __init__(
        self,
        engine_path: str,
        max_batch_size: int = 8,
        categories: List[str] = ["ambulance"],
    ) -> None:
        """
        Initialize ambulance classifier engine.

        Args:
            engine_path (str): path to tensorrt engine file.
            categories (List[str]): list of categories.
            max_batch_size (int): maximum batch size. Defaults to 8.
        """
        super().__init__(engine_path, max_batch_size)
        self.categories = categories

    def predict(
        self, imgs: Union[np.ndarray, List[np.ndarray]], conf: float = 0.25
    ) -> ClsResultSchema:
        """
        Classify cropped image(s).

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): input image(s).
            conf (float): confidence threshold. Defaults to 0.25.

        Returns:
            ClsResultSchema: classification results.
        """
        imgs = self.preprocess_imgs(imgs)

        # iterate to avoid memory error
        outputs: List[HostMemBufferSchema] = []
        for i in range(0, len(imgs), self.max_batch_size):
            batch_imgs = imgs[i : i + self.max_batch_size]
            batch_outputs = self.forward(batch_imgs)
            # TODO: bugs. host needs to be copied
            outputs.extend(
                [
                    HostMemBufferSchema(
                        host=out.host.copy(),
                        device=out.device,
                        binding=out.binding,
                    )
                    for out in batch_outputs
                ]
            )
        results = self.postprocess_outputs(outputs, imgs.shape[0], conf)

        return results

    def preprocess_imgs(self, imgs: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Preprocess image(s) for TensorRT engine."""
        if isinstance(imgs, np.ndarray):
            imgs = [imgs]

        # resize images
        dst_h, dst_w = self.img_shape[2:]
        resized_imgs = np.ones((len(imgs), dst_h, dst_w, 3), dtype=np.float32)
        for i, img in enumerate(imgs):
            resized_imgs[i] = cv2.resize(img, (dst_w, dst_h))

        # normalize and transpose images
        resized_imgs = resized_imgs / 255.0
        resized_imgs = resized_imgs.transpose((0, 3, 1, 2))

        return resized_imgs

    def postprocess_outputs(
        self, outputs: List[HostMemBufferSchema], imgs_shape: int, conf: float = 0.25
    ) -> ClsResultSchema:
        """
        Postprocess classification outputs.

        Args:
            outputs (List[HostMemBufferSchema]): list of classification outputs.
            imgs_shape (int): number of images.
            conf (float): confidence threshold. Defaults to 0.25.

        Returns:
            ClsResultSchema: classification results.
        """
        result = ClsResultSchema()
        for i in range(imgs_shape):
            output = np.apply_along_axis(self.softmax_np, 0, outputs[i].host)
            score = round(output.max(), 2)
            if score > conf:
                category = self.categories[int(np.argmax(output))]
                result.categories.append(category)
                result.scores.append(score)
            else:
                result.categories.append(None)
                result.scores.append(0.0)

        return result

    def softmax_np(self, x: np.ndarray):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
