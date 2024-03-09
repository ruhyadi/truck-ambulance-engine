"""Common TensorRT engine module."""

import rootutils

ROOT = rootutils.autosetup()

from typing import List, Union

import numpy as np
import pycuda.autoinit  # need for initializing CUDA driver
import pycuda.driver as cuda
import tensorrt as trt

from src.schema.trt_schema import HostMemBufferSchema
from src.utils.logger import get_logger

log = get_logger()


class TrtEngine:
    """Common TensorRT engine module."""

    def __init__(
        self,
        engine_path: str,
        max_batch_size: int = 1,
    ) -> None:
        """
        Initialize TensorRT engine.

        Args:
            engine_path (str): Path to TensorRT model file
            max_batch_size (int): Maximum batch size for inference. Defaults to 1.
        """
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size
        self.decrypt_key = None

    def setup(self) -> None:
        """Setup TensorRT engine."""
        log.info(f"Setup Widya GPU AI Engine...")
        logger = trt.Logger(trt.Logger.WARNING)
        logger.min_severity = trt.Logger.Severity.ERROR
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, "")

        # deserialize engine
        with open(str(self.engine_path), "rb") as f:
            engine_data = f.read()
        self.engine = runtime.deserialize_cuda_engine(engine_data)

        # create execution context
        self.img_shape = self.engine.get_binding_shape(0)
        self.context = self.engine.create_execution_context()
        self.inputs: List[HostMemBufferSchema] = []
        self.outputs: List[HostMemBufferSchema] = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            size = (
                trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            )
            size = size * -1 if size < 0 else size  # fix for dynamic batch size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            log.debug(f"Binding '{binding}': {size} {dtype}")
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self.inputs.append(
                    HostMemBufferSchema(
                        host=host_mem, device=device_mem, binding=binding
                    )
                )
            else:
                self.outputs.append(
                    HostMemBufferSchema(
                        host=host_mem, device=device_mem, binding=binding
                    )
                )

        # set context input shape
        self.context.set_binding_shape(0, (self.max_batch_size, *self.img_shape[1:]))
        
        log.debug(
            f"TensorRT engine ready to use. Max input shape: {self.inputs[0].host.reshape(self.img_shape).shape}"
        )

    def forward(
        self, imgs: Union[np.ndarray, List[np.ndarray]]
    ) -> List[HostMemBufferSchema]:
        """
        Forward pass through TensorRT engine.
        
        Args:
            imgs (np.ndarray): Input images. Supports batched input.
    
        Returns:
            List[HostMemBufferSchema]: List of output tensors.
        """
        self.inputs[0].host = np.ravel(imgs)

        # transfer input data to the GPU
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle,
        )

        # transfer predictions back from the GPU
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        # synchronize the stream
        self.stream.synchronize()

        return self.outputs
