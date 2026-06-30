# Video Track Overlay Player

本项目用于把视频里的目标检测/跟踪结果导出成数据，并用前端页面把这些数据叠加回原视频。

实时摄像头检测已迁移到独立的 `rfdetr-live` 项目。

主要用途：

1. 用 RF-DETR 或 YOLO 离线检测并跟踪人、车、鸟、面部等目标。
2. RF-DETR 输出检测框；YOLO `*-seg.pt` 分割模型可输出分割区域。
3. 输出带贴纸的视频，方便检查识别效果。
4. 将原视频副本保存在同一个运行目录，方便整理和对照。
5. 输出 `CSV / JSONL` 数据，方便前端、MIDI、TouchDesigner 或其他视觉系统使用。
6. 用前端播放器加载原视频和跟踪数据，验证图形叠加是否准确。

## 目录结构

```text
video-track-overlay-player/
  source/                    放待检测的原视频
  runs/                      每次运行生成的结果
  scripts/
    track_objects_rfdetr.py    离线 RF-DETR 跟踪、贴纸、数据导出脚本
    track_objects_yolo.py      离线 YOLO 跟踪、贴纸、数据导出脚本
  web_track_overlay_player/  前端跟踪数据叠加播放器
  docs/USAGE.md              详细使用文档，可作为 Wiki 页面源稿
  README.md                  项目简介和安装
  AI_CONTEXT.md              给 AI / Codex 的技术上下文
```

## 推荐环境

建议按用途选择 CPU 或 GPU 环境。

```text
video-track-cpu-env
  CPU 环境，适合通用测试和兜底

video-track-gpu-env
  GPU 环境，适合 NVIDIA CUDA 显卡
```

CPU 环境建议用普通 `venv`。GPU 环境建议用 `conda` 创建 Python 3.11，再安装 CUDA 版 PyTorch。

## 安装

### Windows CPU

前提：

```text
Python 3.10 - 3.13
PowerShell
```

创建并激活环境：

```powershell
python -m venv video-track-cpu-env
.\video-track-cpu-env\Scripts\Activate.ps1
```

如果 PowerShell 拦截激活脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\video-track-cpu-env\Scripts\Activate.ps1
```

安装依赖：

```powershell
pip install -r requirements-cpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证：

```powershell
python -c "from ultralytics import YOLO; import cv2; import torch; print('ok'); print(torch.cuda.is_available())"
```

CPU 环境看到 `False` 是正常的。

退出 CPU 环境：

```powershell
deactivate
```

### Windows NVIDIA GPU

适合 NVIDIA CUDA 显卡。推荐用 conda 创建 Python 3.11 环境，避免 CUDA 版 PyTorch 和 Python 版本不匹配。

前提：

```text
Miniconda 或 Anaconda
NVIDIA 显卡驱动已安装
```

创建并激活环境：

```powershell
conda create -p .\video-track-gpu-env python=3.11 -y
conda activate .\video-track-gpu-env
```

安装 CUDA 版 PyTorch 和项目依赖：

```powershell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu-cu121.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果使用 RF-DETR 离线检测，再安装：

```powershell
pip install -r requirements-rfdetr.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证 GPU：

```powershell
python -c "from ultralytics import YOLO; import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

如果输出 `True` 和显卡名称，GPU 环境可用。

退出 GPU 环境：

```powershell
conda deactivate
```

### macOS

macOS 默认按 CPU 环境使用。Intel Mac 没有 CUDA，Apple Silicon 的 `mps` 后端兼容性不如 CUDA 稳定，不作为默认安装方式。

创建并激活环境：

```bash
python3 -m venv video-track-cpu-env
source video-track-cpu-env/bin/activate
```

安装依赖：

```bash
pip install -r requirements-cpu.txt
```

验证：

```bash
python -c "from ultralytics import YOLO; import cv2; import torch; print('ok'); print(torch.cuda.is_available())"
```

退出 macOS 环境：

```bash
deactivate
```

Apple Silicon 用户可以自行尝试 `mps`：

```bash
python -c "import torch; print(torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

## 快速入口

详细用法见 [docs/USAGE.md](docs/USAGE.md)。

RF-DETR 离线处理视频：

```powershell
python scripts\track_objects_rfdetr.py --classes car --device cuda
```

RF-DETR 离线分割视频：

```powershell
python scripts\track_objects_rfdetr.py --model-size seg-small --classes person --device cuda
```

YOLO 离线处理视频：

```powershell
python scripts\track_objects_yolo.py
```

YOLO 离线分割视频：

```powershell
python scripts\track_objects_yolo.py --model yolo11n-seg.pt --classes person
```

YOLO 离线检测面部：

```powershell
python scripts\track_objects_yolo.py --model yolov11n-face.pt --classes 0
```

前端离线叠加播放器：

```powershell
# Windows CPU（venv 环境）
cd web_track_overlay_player
..\video-track-cpu-env\Scripts\python.exe -m http.server 8090 --bind 127.0.0.1

# Windows GPU（conda 环境）
cd web_track_overlay_player
..\video-track-gpu-env\python.exe -m http.server 8090 --bind 127.0.0.1
```

打开：

```text
http://127.0.0.1:8090
```
