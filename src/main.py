"""
Main detection function.
usage:
python src/main.py \
    +input_file=tmp/sample001.png \
    +output_path=tmp/outputs
"""

import rootutils

ROOT = rootutils.autosetup()

from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig

from src.utils.logger import get_logger
from src.utils.plotter import PlotUtils

log = get_logger()


def detection(cfg: DictConfig) -> None:
    """Detection main function."""
    log.info(f"Start detection...")

    # additional configs
    conf = 0.25 if "conf" not in cfg else cfg.conf
    nms = 0.45 if "nms" not in cfg else cfg.nms
    input_file = Path(cfg.input_file)
    output_path = Path(cfg.output_path) if "output_path" in cfg else Path("tmp/outputs")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / input_file.name

    # setup plotter
    plotter = PlotUtils()

    # setup engine
    if cfg.engine.yolo.engine_path.endswith(".plan"):
        from src.engine.yolo_trt_engine import YoloTrtEngine

        engine = YoloTrtEngine(**cfg.engine.yolo)
        engine.setup()
    elif cfg.engine.yolo.engine_path.endswith(".onnx"):
        from src.engine.yolo_onnx_engine import YoloOnnxEngine

        engine = YoloOnnxEngine(**cfg.engine.yolo)
        engine.setup()
    else:
        raise ValueError(f"Unknown engine type: {cfg.engine.yolo.engine_path}")

    # run detection
    img = cv2.imread(str(input_file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plotter.setup(img)

    result = engine.predict(imgs=[img], conf=conf, nms=nms)[0]

    # draw predictions
    img = plotter.draw_boxes(
        frame=img,
        boxes=result.boxes,
        labels=result.categories,
    )

    # save output
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_file), img)

    log.info(f"Detection done. Output: {output_file}")


if __name__ == "__main__":
    """Main function."""

    @hydra.main(config_path=f"{ROOT}/configs", config_name="main", version_base=None)
    def main(cfg: DictConfig) -> None:
        """Main function."""
        detection(cfg)

    main()
