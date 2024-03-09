"""YOLO TensorRT engine module."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

import cv2
import numpy as np

from src.engine.trt_engine import TrtEngine
from src.schema.trt_schema import End2EndResultSchema, HostMemBufferSchema
from src.schema.yolo_schema import YoloResultSchema
from src.utils.logger import get_logger
from src.utils.nms_utils import multiclass_nms

log = get_logger()


class YoloTrtEngine(TrtEngine):
    """YOLO TensorRT engine module."""

    def __init__(
        self,
        engine_path: str,
        max_batch_size: int,
        categories: List[str],
        end2end: bool = False,
        arch: str = "yolox",
        pretrained: bool = False,
        max_det_end2end: int = 100,
    ) -> None:
        """
        Initialize YOLO TensorRT engine.

        Args:
            engine_path (str): Path to TensorRT model file.
            max_batch_size (int): Maximum batch size for inference.
            categories (List[str]): List of categories.
            end2end (bool): End-to-end inference flag.
            arch (str): YOLO architecture. Defaults to "yolox".
            pretrained (bool): Whether the model is pretrained (COCO with 80 classes).
            max_det_end2end (int): Maximum number of detections in end-to-end mode.
        """
        assert arch in ["yolox", "yolov8"], "Invalid architecture"
        super().__init__(engine_path, max_batch_size)
        self.categories = categories
        self.end2end = end2end
        self.arch = arch
        self.pretrained = pretrained
        self.max_det_end2end = max_det_end2end

        self.normalize = False if self.arch == "yolox" else True

    def predict(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        conf: float = 0.25,
        nms: float = 0.45,
    ) -> List[YoloResultSchema]:
        """
        Predict detection on image(s).

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): Image(s) to predict.
            conf (float, optional): Confidence threshold. Defaults to 0.25.
            nms (float, optional): NMS threshold. Defaults to 0.45.

        Returns:
            List[YoloResultSchema]: List of detection results.
        """
        imgs, ratios, pads = self.preprocess_imgs(imgs, normalize=self.normalize)
        outputs = self.forward(imgs)
        if self.end2end:
            results = self.postprocess_end2end(outputs, ratios, pads, conf)
        else:
            results = self.postprocess_nms(outputs, ratios, pads, conf, nms)

        return results

    def preprocess_imgs(
        self,
        imgs: Union[np.ndarray, List[np.ndarray]],
        mode: str = "center",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Preprocess image(s) (batch) like resize, normalize, padding, etc.

        Args:
            imgs (Union[np.ndarray, List[np.ndarray]]): Image(s) to preprocess.
            mode (str, optional): Padding mode. Defaults to "center".
            normalize (bool, optional): Whether to normalize image(s). Defaults to True.

        Returns:
            np.ndarray: Preprocessed image(s) in size (B, C, H, W).
        """
        assert mode in ["center", "left"]
        if isinstance(imgs, np.ndarray):
            # convert PNG to JPEG
            if imgs.shape[-1] == 4:
                imgs = cv2.cvtColor(imgs, cv2.COLOR_RGBA2RGB)
            imgs = [imgs]

        # resize and pad
        dst_h, dst_w = self.img_shape[2:]
        resized_imgs = np.ones((len(imgs), dst_h, dst_w, 3), dtype=np.float32) * 114
        ratios = np.ones((len(imgs)), dtype=np.float32)
        pads = np.ones((len(imgs), 2), dtype=np.float32)
        for i, img in enumerate(imgs):
            src_h, src_w = img.shape[:2]
            ratio = min(dst_w / src_w, dst_h / src_h)
            resized_w, resized_h = int(src_w * ratio), int(src_h * ratio)
            dw, dh = (dst_w - resized_w) / 2, (dst_h - resized_h) / 2
            img = cv2.resize(img, (resized_w, resized_h))
            if mode == "center":
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
                img = cv2.copyMakeBorder(
                    img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=114
                )
                resized_imgs[i] = img
            elif mode == "left":
                resized_imgs[i][:resized_h, :resized_w, :] = img

            pads[i] = np.array([dw, dh], dtype=np.float32)
            ratios[i] = ratio

        # normalize
        resized_imgs = resized_imgs.transpose(0, 3, 1, 2)
        resized_imgs /= 255.0 if normalize else 1.0
        # resized_imgs = np.ascontiguousarray(resized_imgs).astype(np.float32)

        return resized_imgs, ratios, pads

    def postprocess_end2end(
        self,
        outputs: List[HostMemBufferSchema],
        ratios: np.ndarray,
        pads: np.ndarray,
        conf: float = 0.25,
    ) -> List[YoloResultSchema]:
        """
        Postprocess outputs in end-to-end mode.

        Args:
            outputs (List[HostMemBufferSchema]): Model outputs.
            ratios (np.ndarray): Resizing ratios.
            pads (np.ndarray): Padding values.
            conf (float, optional): Confidence threshold. Defaults to 0.25.

        Returns:
            List[YoloResultSchema]: List of detection results.
        """
        end2end_output: List[End2EndResultSchema] = []
        for i in range(len(ratios)):  # iterate over imgs in batch
            # NOTE: assume bbox have 4 * 100 candidates score have 100 candidates, and index have 100 candidates
            num_det = outputs[0].host[i]
            bbox = outputs[1].host[
                (i * 4 * self.max_det_end2end) : ((i + 1) * 4 * self.max_det_end2end)
            ]
            score = outputs[2].host[
                (i * self.max_det_end2end) : ((i + 1) * self.max_det_end2end)
            ]
            class_index = outputs[3].host[
                (i * self.max_det_end2end) : ((i + 1) * self.max_det_end2end)
            ]
            end2end_output.append(
                End2EndResultSchema(
                    num_det=num_det,
                    boxes=bbox,
                    scores=score,
                    categories=class_index,
                )
            )

        # scaling and filtering
        results: List[YoloResultSchema] = []
        for i, out in enumerate(end2end_output):
            num_det = int(out.num_det)
            bbox = out.boxes[: num_det * 4].reshape(-1, 4)
            score = out.scores[:num_det]
            class_index = out.categories[:num_det]

            # scale bbox to original image size
            bbox[:, 0] -= pads[i, 0]
            bbox[:, 1] -= pads[i, 1]
            bbox[:, 2] -= pads[i, 0]
            bbox[:, 3] -= pads[i, 1]
            bbox /= ratios[i]

            # filter by confidence and class
            mask_score = score > conf
            mask_class = class_index < len(self.categories)
            bbox: np.ndarray = bbox[mask_score & mask_class]
            score = score[mask_score & mask_class]
            class_index = class_index[mask_score & mask_class]

            if len(bbox) == 0:
                results.append(YoloResultSchema())
                continue

            results.append(
                YoloResultSchema(
                    boxes=bbox.astype(np.int32).tolist(),
                    scores=score.tolist(),
                    categories=[self.categories[i] for i in class_index.tolist()],
                )
            )

        return results

    def postprocess_nms(
        self,
        outputs: List[HostMemBufferSchema],
        ratios: np.ndarray,
        pads: np.ndarray,
        conf: float = 0.25,
        nms: float = 0.45,
    ) -> List[YoloResultSchema]:
        """
        Postprocess outputs with NMS.

        Args:
            outputs (List[HostMemBufferSchema]): Model outputs.
            ratios (np.ndarray): Resizing ratios.
            pads (np.ndarray): Padding values.
            conf (float, optional): Confidence threshold. Defaults to 0.25.
            nms (float, optional): NMS threshold. Defaults to 0.45.

        Returns:
            List[YoloResultSchema]: List of detection results.
        """
        # reshape outputs to (batch, -1, num_classes + 5)
        batch_size = ratios.shape[0]
        num_classes = len(self.categories) if not self.pretrained else 80
        outputs: np.ndarray = outputs[0].host.reshape(
            self.max_batch_size, -1, (num_classes + 5)
        )
        outputs = outputs[:batch_size]

        # get boxes, scores, classes
        boxes = outputs[:, :, :4]
        classes = outputs[:, :, 5:].argmax(axis=-1)
        scores = outputs[:, :, 4:5] * outputs[:, :, 5:]

        # convert xywh to xyxy
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, :, 0] = boxes[:, :, 0] - boxes[:, :, 2] / 2
        boxes_xyxy[:, :, 1] = boxes[:, :, 1] - boxes[:, :, 3] / 2
        boxes_xyxy[:, :, 2] = boxes[:, :, 0] + boxes[:, :, 2] / 2
        boxes_xyxy[:, :, 3] = boxes[:, :, 1] + boxes[:, :, 3] / 2

        # scaling boxes to original image size
        boxes_xyxy[:, :, 0] -= pads[:, 0, None]
        boxes_xyxy[:, :, 1] -= pads[:, 1, None]
        boxes_xyxy[:, :, 2] -= pads[:, 0, None]
        boxes_xyxy[:, :, 3] -= pads[:, 1, None]
        boxes_xyxy /= ratios[:, None, None]

        # nms
        results: List[YoloResultSchema] = []
        for i in range(batch_size):
            dets = multiclass_nms(
                boxes=boxes_xyxy[i],
                scores=scores[i],
                nms=nms,
                conf=conf,
            )
            if dets is None:
                results.append(YoloResultSchema())
                continue

            # filter confidence and class
            mask_score = dets[:, -2] > conf
            mask_class = dets[:, -1] < len(self.categories)
            dets = dets[mask_score & mask_class]

            # get class names
            class_ids = dets[:, -1].astype(np.int32).tolist()
            class_names = [self.categories[int(i)] for i in class_ids]

            # filter by class names. Ignore if class_names is "ignore"
            if "ignore" in self.categories:
                mask_ignore = np.array(
                    [cls not in ["ignore"] for cls in class_names], dtype=np.bool
                )
                dets = dets[mask_ignore]
                class_names = [cls for cls in class_names if cls != "ignore"]

            results.append(
                YoloResultSchema(
                    categories=class_names,
                    scores=dets[:, -2].tolist(),
                    boxes=dets[:, :-2].astype(np.int32).tolist(),
                )
            )

        return results


if __name__ == "__main__":
    """Debugging."""

    engine = YoloTrtEngine(
        engine_path="tmp/models/yolov8_s_trt8.plan",
        max_batch_size=1,
        categories=[str(i) for i in range(80)],
        end2end=True,
        arch="yolov8",
        pretrained=True,
        max_det_end2end=100,
    )
    engine.setup()

    img = cv2.imread("tmp/sample001.png")
    results = engine.predict(img)

    for res in results:
        for box, score, category in zip(res.boxes, res.scores, res.categories):
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{category} {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    log.warning(f"Result: {results}")

    cv2.imwrite(f"tmp/sample001_yolov8_trt8.png", img)