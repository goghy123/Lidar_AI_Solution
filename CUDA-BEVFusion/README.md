# CUDA-BEVFusion

This repository contains sources and model for [BEVFusion](https://github.com/mit-han-lab/bevfusion) inference using CUDA & TensorRT.
![title](/assets/bevfusion.png)


## 3D Object Detection(on nuScenes validation set)
- For all models, we used the [BEVFusion-Base](https://github.com/mit-han-lab/bevfusion/blob/main/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml) configuration.
  - The camera resolution is 256x704
- For the camera backbone, we chose SwinTiny and [ResNet50](configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/default.yaml).

|         **Model**        | **Framework** | **Precision** | **mAP** | **NDS** | **FPS** |
|:------------------------:|:-------------:|:-------------:|:-------:|:-------:|:----------------:|
| Swin-Tiny <br/> BEVFusion-Base |    PyTorch    |   FP32+FP16   |  68.52  |  71.38  |         8.4(on RTX3090)        |
|         ResNet50         |    PyTorch    |   FP32+FP16   |  67.93  |  70.97  |         -        |
|         ResNet50         |    TensorRT   |      FP16     |  67.89  |  70.98  |        18(on ORIN)        |
|         ResNet50-PTQ         |    TensorRT   |      FP16+INT8     |  67.66  |  70.81  |        25(on ORIN)        |
- Note: The time we reported on ORIN is based on the average of nuScenes 6019 validation samples.
  - Since the number of lidar points is the main reason that affects the FPS. 
  - Please refer to the readme of [3DSparseConvolution](/libraries/3DSparseConvolution/README.md) for more details.

## Demonstration
![](../assets/cuda-bevfusion.gif)

## Model and Data
- For quick practice, we provide an example data of nuScenes. You can download it from ( [NVBox](https://nvidia.box.com/shared/static/g8vxxes3xj1288teyo4og87rn99brdf8) ) or ( [Baidu Drive](https://pan.baidu.com/s/1ED6eospSIF8oIQ2unU9WIQ?pwd=mtvt) ). It contains the following:
  1. Camera images in 6 directions.
  2. Transformation matrix of camera/lidar/ego.
  3. Use for bevfusion-pytorch data of example-data.pth, allow export onnx only without depending on the full dataset.
- All models (model.zip) can be downloaded from ( [NVBox](https://nvidia.box.com/shared/static/vc1ezra9kw7gu7wg3v8cwuiqshwr8b39) ) or ( [Baidu Drive](https://pan.baidu.com/s/1BiAoQ8L7nC45vEwkN3bSGQ?pwd=8jb6) ). It contains the following:
  1. swin-tiny onnx models.
  2. resnet50 onnx and pytorch models.
  3. resnet50 int8 onnx and PTQ models.

## Prerequisites
To build bevfusion, we need to depend on the following libraries:
- CUDA >= 11.0
- CUDNN >= 8.2
- TensorRT >= 8.5.0
- libprotobuf-dev
- [Compute Capability](https://developer.nvidia.com/cuda-gpus#compute) >= sm_80
- Python >= 3.6

The data in the performance table was obtained by us on the Nvidia Orin platform, using TensorRT-8.6, cuda-11.4 and cudnn8.6 statistics.

## Quick Start for Inference
- note: Please use `git clone --recursive` to pull this repository to ensure the integrity of the dependencies.

### 1. Download models and datas to CUDA-BEVFusion directory
- download model.zip from ( [NVBox](https://nvidia.box.com/shared/static/vc1ezra9kw7gu7wg3v8cwuiqshwr8b39) ) or ( [Baidu Drive](https://pan.baidu.com/s/1_6IJTzKlJ8H62W5cUPiSbA?pwd=g6b4) )
- download nuScenes-example-data.zip from 
( [NVBox](https://nvidia.box.com/shared/static/g8vxxes3xj1288teyo4og87rn99brdf8) ) or ( [Baidu Drive](https://pan.baidu.com/s/1ED6eospSIF8oIQ2unU9WIQ?pwd=mtvt) )
```bash
# download models and datas to CUDA-BEVFusion
cd CUDA-BEVFusion

# unzip models and datas
unzip model.zip
unzip nuScenes-example-data.zip

# here is the directory structure after unzipping
CUDA-BEVFusion
|-- example-data
    |-- 0-FRONT.jpg
    |-- 1-FRONT_RIGHT.jpg
    |-- ...
    |-- camera_intrinsics.tensor
    |-- ...
    |-- example-data.pth
    `-- points.tensor
|-- src
|-- qat
|-- model
    |-- resnet50int8
    |   |-- bevfusion_ptq.pth
    |   |-- camera.backbone.onnx
    |   |-- camera.vtransform.onnx
    |   |-- default.yaml
    |   |-- fuser.onnx
    |   |-- head.bbox.onnx
    |   `-- lidar.backbone.xyz.onnx
    |-- resnet50
    `-- swint
|-- bevfusion
`-- tool
```
### 2. Configure the environment.sh
- Install python dependency libraries
```bash
apt install libprotobuf-dev
pip install onnx
```

- Modify the TensorRT/CUDA/CUDNN/BEVFusion variable values in the tool/environment.sh file.
```bash
# change the path to the directory you are currently using
export TensorRT_Lib=/path/to/TensorRT/lib
export TensorRT_Inc=/path/to/TensorRT/include
export TensorRT_Bin=/path/to/TensorRT/bin

export CUDA_Lib=/path/to/cuda/lib64
export CUDA_Inc=/path/to/cuda/include
export CUDA_Bin=/path/to/cuda/bin
export CUDA_HOME=/path/to/cuda

export CUDNN_Lib=/path/to/cudnn/lib

# For CUDA-11.x:    SPCONV_CUDA_VERSION=11.4
# For CUDA-12.x:    SPCONV_CUDA_VERSION=12.6
export SPCONV_CUDA_VERSION=11.4

# resnet50/resnet50int8/swint
export DEBUG_MODEL=resnet50int8

# fp16/int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=example-data
export USE_Python=OFF
```

- Apply the environment to the current terminal.
```bash
. tool/environment.sh
```

### 5. Compile and run

1. Building the models for tensorRT
```bash
bash tool/build_trt_engine.sh
```

2. Compile and run the program
```bash
# Generate the protobuf code
bash src/onnx/make_pb.sh

# Compile and run
bash tool/run.sh
```

### 使用 nuScenes Mini 多帧数据运行 CUDA-BEVFusion

#### 1. 数据集放置

将解压后的 nuScenes mini 数据集放到 `CUDA-BEVFusion` 目录下：

```bash
CUDA-BEVFusion/
└── nuscenesmini/
    ├── maps
    ├── samples
    ├── sweeps
    └── v1.0-mini
```

#### 2. 数据集查看

在 CUDA-BEVFusion 根目录运行：

```BASH
python - <<'PY'
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version='v1.0-mini',
    dataroot='./nuscenesmini',
    verbose=False
)

print('Scenes:', len(nusc.scene))
print('Samples:', len(nusc.sample))

for i, scene in enumerate(nusc.scene):
    print(
        i,
        scene['name'],
        'samples:',
        scene['nbr_samples'],
        '|',
        scene['description']
    )
PY
```

#### 3. 转换数据格式

CUDA-BEVFusion 不直接读取 nuScenes 原始目录，需要先转换成与 example-data 相同的输入格式。

执行转换：

``` BASH
python tool/tools_convert_nuscenes_mini.py \
  --dataroot ./nuscenesmini \
  --version v1.0-mini \
  --out nuscenesmini-frames \
  --max-samples 20 \
  --scene-index 0
```

若要转换mini数据集全部关键帧：

``` BASH
for i in $(seq 0 9); do
    echo "Converting scene $i"

    python tool/tools_convert_nuscenes_mini.py \
      --dataroot ./nuscenesmini \
      --version v1.0-mini \
      --out nuscenesmini-frames/scene-$i \
      --max-samples 999 \
      --scene-index $i
done
```

#### 4. 运行

也可以通过修改 tool/environment.sh中的参数

``` SH
# 单帧运行：

export DEBUG_MODEL=resnet50int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=nuscenes-mini-frames/frame_000000
export DEBUG_OUTPUT_DIR=nuscenes-mini-output-single
bash tool/run.sh
```

```BASH
# 多帧运行：

export DEBUG_MODEL=resnet50int8
export DEBUG_PRECISION=int8
export DEBUG_DATA=dataset/nuscenesmini-frames
export DEBUG_OUTPUT_DIR=nuscenes-mini-outputs-maincpp
bash tool/run.sh
```

```BASH
# 批量运行：

for i in $(seq 0 9); do
    echo "Processing scene $i"
    export DEBUG_MODEL=resnet50int8
    export DEBUG_PRECISION=int8
    export DEBUG_DATA=dataset/nuscenesmini-frames/scene-$i
    export DEBUG_OUTPUT_DIR=outputs/scene-$i

    bash tool/run.sh

done
```

#### 5. 将输出图片拼接成 GIF 或视频

CUDA-BEVFusion 批量运行后，每一帧的可视化结果通常会保存为连续图片，例如：

```bash
outputs/scene-0/
├── frame_000000.jpg
├── frame_000001.jpg
├── frame_000002.jpg
└── ...
```

使用 tool/images_to_gif_or_video.py 将这些连续帧拼接成 GIF 动图或 MP4 视频。

脚本默认参数可以在文件开头修改,不指定终端参数时，直接运行：

``` BASH
python tool/images_to_gif_or_video.py
```

也可以在终端指定输入目录、输出目录、输出类型和帧率：

```BASH
# 生成视频：
python tool/images_to_gif_or_video.py \
  --input-dir outputs/scene-0 \
  --output-dir outputs/video \
  --output-type video \
  --output-name scene-0 \
  --fps 7
```

```BASH
# 生成GIF动图：
python tool/images_to_gif_or_video.py \
  --input-dir outputs/scene-0 \
  --output-dir outputs/gif \
  --output-type gif \
  --output-name scene-0 \
  --fps 7
```

```BASH
# 批量将多个 scene 的输出结果转换为视频：
for i in $(seq 0 9); do
    echo "Making video for scene $i"
    python tool/images_to_gif_or_video.py \
      --input-dir outputs/scene-$i \
      --output-dir outputs/videos \
      --output-type video \
      --output-name scene-$i \
      --fps 7
done
```

```BASH
# 查看所有可用参数：
python tool/images_to_gif_or_video.py --help
```

## Export onnx and PTQ
- For more detail, please refer [here](qat/README.md)

## For Python Interface
1. Modify `USE_Python=ON` in environment.sh to enable compilation of python.
2. Run `bash tool/run.sh` to build the libpybev.so.
3. Run `python tool/pybev.py` to test the python interface.

## For PyTorch BEVFusion
- Use the following command to get a specific commit to avoid failure.
```bash
git clone https://github.com/mit-han-lab/bevfusion

cd bevfusion
git checkout db75150717a9462cb60241e36ba28d65f6908607
```

## Further performance improvement
- Since the number of point clouds fluctuates more, this has a significant impact on the FPS.
  - Consider using the ground removal or range filter algorithms provided in [cuPCL](https://github.com/NVIDIA-AI-IOT/cuPCL), which can decrease the inference time by lidar.
- We just implemented the recommended partial quantization method. However, users can further reduce the inference latency by sparse pruning and 4:2 sparsity.
  - In the resnet50 model at large resolutions, using the --sparsity=force option can significantly improve inference performance. For more details, please refer to [ASP](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity) (automatic sparsity tools).
- In general, the camera backbone has less impact on accuracy and more impact on latency.
  - A lighter camera backbone (such as resnet34) will achieve lower latency.

## References
- [BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation](https://arxiv.org/abs/2205.13542)
- [BEVFusion Repository](https://github.com/mit-han-lab/bevfusion)


