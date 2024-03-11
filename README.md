# Truck-Ambulance Engine

## Introduction
Truck-ambulance engine is an AI engine built on top of the [TensorRT](https://developer.nvidia.com/tensorrt) and [ONNX Runtime](https://onnxruntime.ai/) libraries to perform real-time object detection of trucks and ambulances in images, videos, and live camera feeds. 

## Getting Started
### Prerequisites
We assume you have **docker** installed on you machine. If not, you can install it from [here](https://docs.docker.com/get-docker/). Also, you need to have installed the **NVIDIA Container Toolkit** to run the docker container with GPU support. You can install it from [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Development
For development, we will use [devcontainer](https://code.visualstudio.com/docs/devcontainers/containers) to create a development environment. So, you need to have **Visual Studio Code** installed on your machine. If not, you can install it from [here](https://code.visualstudio.com/). Also, you need to have the **Remote - Containers** extension installed on your Visual Studio Code. If not, you can install it from [here](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

Next, you need to clone the repository to your local machine by:
```bash
git clone https://github.com/ruhyadi/truck-ambulance-engine
```
There is two type of devcontainer you can choose: `gpu-devel` and `cpu-devel`. If you have a GPU, you can choose `gpu-devel` and if you don't have a GPU, you can choose `cpu-devel`. You can choose the devcontainer by pressing `F1` and type `Remote-Containers: Rebuild Container` (or `Reopen in Container`). Then, choose the devcontainer you want to use.

### Production


## Acknowledgements
- [TensorRT](https://developer.nvidia.com/tensorrt): TensorRT is a high-performance deep learning inference library for production environments. It is built to optimize and deploy deep learning models on NVIDIA GPUs.
- [ONNX Runtime](https://onnxruntime.ai/): ONNX Runtime is a high-performance scoring engine for Open Neural Network Exchange (ONNX) models. It is optimized for both cloud and edge and works on Linux, Windows, and Mac.