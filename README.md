# YOLO Track Overlay Player

本项目用于把视频里的目标检测/跟踪结果导出成数据，并用前端页面把这些数据重新叠加到原视频上。

当前用途：

1. 用 YOLO 检测并跟踪视频中的人、车、鸟等目标。
2. 输出带贴纸的视频，方便检查识别效果。
3. 输出 `CSV / JSONL` 数据，方便后续前端、MIDI、TouchDesigner 或其他视觉系统使用。
4. 用前端播放器加载原视频和检测数据，验证图形叠加是否准确。

## 目录结构

```text
yolo-track-overlay-player/
  source/                    放待检测的原视频
  runs/                      每次运行生成的结果
  web_track_overlay_player/  前端跟踪数据叠加播放器
  scripts/track_objects_stickers.py  YOLO 跟踪、贴纸、数据导出脚本
  README.md                  项目说明与使用文档
  AI_CONTEXT.md              给 AI / Codex 的技术上下文
```

## 推荐环境

建议按用途选择 CPU 或 GPU 环境。

```text
yolo-cpu-env
  CPU 环境，适合通用测试和兜底

yolo-gpu-env
  GPU 环境，适合 NVIDIA CUDA 显卡
```

CPU 环境建议用普通 `venv`。GPU 环境建议用 `conda` 创建 Python 3.11，再安装 CUDA 版 PyTorch。

## 安装

### Windows CPU

适合没有 NVIDIA 显卡，或者只是先验证流程的电脑。

前提：

```text
Python 3.10 - 3.13
PowerShell
```

在项目根目录打开 PowerShell，创建环境：

```powershell
python -m venv yolo-cpu-env
```

激活环境：

```powershell
.\yolo-cpu-env\Scripts\Activate.ps1
```

如果 PowerShell 拦截激活脚本：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\yolo-cpu-env\Scripts\Activate.ps1
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

### Windows NVIDIA GPU

适合 NVIDIA CUDA 显卡。推荐用 conda 创建 Python 3.11 环境，避免 CUDA 版 PyTorch 和 Python 版本不匹配。

前提：

```text
Miniconda 或 Anaconda
NVIDIA 显卡驱动已安装
```

在项目根目录打开 PowerShell，创建环境：

```powershell
conda create -p .\yolo-gpu-env python=3.11 -y
```

激活环境：

```powershell
conda activate .\yolo-gpu-env
```

安装 CUDA 版 PyTorch。先让 PyTorch 三件套走官方 cu121 wheel 源，再用清华 PyPI 补普通依赖：

```powershell
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-gpu-cu121.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装 Ultralytics 和跟踪依赖：

```powershell
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证 GPU：

```powershell
python -c "from ultralytics import YOLO; import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

如果输出 `True` 和显卡名称，GPU 环境可用。

### macOS

macOS 默认按 CPU 环境使用。Intel Mac 没有 CUDA，Apple Silicon 的 `mps` 后端兼容性不如 CUDA 稳定，不作为默认安装方式。

在项目根目录打开终端，创建环境：

```bash
python3 -m venv yolo-cpu-env
```

激活环境：

```bash
source yolo-cpu-env/bin/activate
```

安装依赖：

```bash
pip install -r requirements-cpu.txt
```

验证：

```bash
python -c "from ultralytics import YOLO; import cv2; import torch; print('ok'); print(torch.cuda.is_available())"
```

Apple Silicon 用户可以自行尝试 `mps`：

```bash
python -c "import torch; print(torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)"
```

如果 `mps` 可用，也建议先按 CPU 路线完成安装和验证。后续运行时再按需要尝试 `--device mps`。

## 使用

本章中的检测命令都需要先激活对应的 CPU 或 GPU 环境。

### CPU

Windows 在项目根目录打开 PowerShell，激活 CPU 环境：

```powershell
.\yolo-cpu-env\Scripts\Activate.ps1
```

运行默认检测：

```powershell
python scripts\track_objects_stickers.py
```

macOS 在项目根目录打开终端，激活 CPU 环境：

```bash
source yolo-cpu-env/bin/activate
```

运行默认检测：

```bash
python scripts/track_objects_stickers.py
```

默认会：

- 读取 `source/` 中最新的 `.mp4`
- 使用 `yolo11n.pt`
- 检测 `car`
- 使用 `conf=0.15`
- 输出到 `runs/`

### GPU

在项目根目录打开 PowerShell。

激活 GPU 环境：

```powershell
conda activate .\yolo-gpu-env
```

确认 GPU 可用：

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

应该看到：

```text
True
NVIDIA GPU name
```

用 GPU 跑检测：

```powershell
python scripts\track_objects_stickers.py --device 0
```

用更细致的模型跑人：

```powershell
python scripts\track_objects_stickers.py --model yolo11m.pt --classes person --device 0
```

### 放入待检测视频

把视频放到：

```text
source/
```

脚本默认会选择 `source/` 里最后修改时间最新的 `.mp4`。

如果要指定某个视频：

Windows PowerShell：

```powershell
python scripts\track_objects_stickers.py --source source\your_video.mp4
```

macOS 终端：

```bash
python scripts/track_objects_stickers.py --source source/your_video.mp4
```

### 常用检测命令

检测车：

```powershell
python scripts\track_objects_stickers.py --classes car
```

检测人：

```powershell
python scripts\track_objects_stickers.py --classes person
```

同时检测人和车：

```powershell
python scripts\track_objects_stickers.py --classes person,car
```

检测鸟：

```powershell
python scripts\track_objects_stickers.py --classes bird
```

降低置信度，减少漏检：

```powershell
python scripts\track_objects_stickers.py --classes person --conf 0.1
```

使用更大的模型：

```powershell
python scripts\track_objects_stickers.py --model yolo11l.pt --classes person --device 0
```

每一帧都显示进度：

```powershell
python scripts\track_objects_stickers.py --progress-every 1
```

关闭进度显示：

```powershell
python scripts\track_objects_stickers.py --progress-every 0
```

### 支持的类别

脚本内置了一些常用别名：

```text
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, bird, cat, dog, horse, sheep, cow
```

也可以直接用 COCO 类别 ID：

```powershell
python scripts\track_objects_stickers.py --classes 0,2
```

常用 ID：

```text
0  = person
1  = bicycle
2  = car
3  = motorcycle
5  = bus
7  = truck
14 = bird
15 = cat
16 = dog
```

### 模型选择

```text
yolo11n.pt  最快，容易漏检
yolo11s.pt  小模型，比 n 稳
yolo11m.pt  中模型，精度和速度折中
yolo11l.pt  大模型，更准更慢
yolo11x.pt  最大，最慢
```

建议顺序：

```text
n -> s -> m -> l -> x
```

CPU 环境建议用 `n / s`。GPU 环境可以尝试 `m / l / x`。

### 退出环境

退出 CPU 环境：

```powershell
deactivate
```

退出 GPU 环境：

```powershell
conda deactivate
```

### 输出结果在哪里

每次运行都会在 `runs/` 下新建文件夹。

例子：

```text
runs/yolo11m_person_gpu_conf0p15_20260521_180000/
  your_video_stickers.mp4
  your_video_tracks.csv
  your_video_tracks.jsonl
```

文件夹名含义：

```text
模型_类别_cpu或gpu_置信度_时间戳
```

文件说明：

```text
*_stickers.mp4   带检测框和贴纸的视频
*_tracks.csv     表格数据，适合 Excel / WPS 查看
*_tracks.jsonl   逐帧 JSON 数据，适合程序读取
```

### CSV / JSONL 字段

`tracks.csv` 每一行是一帧里的一个目标。

主要字段：

```text
frame             第几帧
track_id          跟踪 ID，同一个目标会尽量保持同一个 ID
class_id          COCO 类别 ID
class_name        类别名
confidence        置信度
x1, y1, x2, y2    检测框坐标，单位是原视频像素
center_x/y        中心点坐标，单位是原视频像素
center_x/y_norm   归一化中心点，范围 0-1
```

`tracks.jsonl` 是一行一帧：

```json
{"frame": 1, "objects": [...]}
{"frame": 2, "objects": [...]}
{"frame": 3, "objects": [...]}
```

使用 `JSONL` 的原因：

- 视频数据本来就是逐帧的
- 长视频不需要一次性读取巨大 JSON
- 适合后续做实时播放、MIDI 映射、前端逐帧同步

### 前端叠加播放器

前端页面在：

```text
web_track_overlay_player/
```

不要直接双击 `index.html`。浏览器的 `file://` 安全限制可能导致拖拽视频失败。

启动本地服务器：

Windows PowerShell：

```powershell
cd web_track_overlay_player
..\yolo-cpu-env\Scripts\python.exe -m http.server 8090 --bind 127.0.0.1
```

macOS 终端：

```bash
cd web_track_overlay_player
../yolo-cpu-env/bin/python -m http.server 8090 --bind 127.0.0.1
```

打开：

```text
http://127.0.0.1:8090
```

使用方式：

1. 拖入原始 `.mp4` 视频。
2. 拖入对应的 `*_tracks.jsonl`。
3. 点击 `Play` 播放。
4. 调整 FPS、透明度、图形类型。

播放器逻辑：

- 原视频填满浏览器窗口
- 视频比例不变，多余部分裁切
- canvas 使用同样的 cover 映射
- 因此检测框会跟随视频缩放和裁切

## 常见问题

`torch.cuda.is_available()` 是 `False`

通常是装到了 CPU 版 PyTorch，或者 Python 版本不匹配 CUDA wheel。GPU 环境建议使用 Python 3.11。

`lap` 缺失

跟踪模式需要 `lap`：

```powershell
pip install lap -i https://pypi.tuna.tsinghua.edu.cn/simple
```

检测不到目标

按顺序尝试：

1. 降低 `--conf`
2. 换更大的模型
3. 确认 `--classes` 写对
4. 使用 GPU 后再尝试更大模型

拖拽视频到前端时报 `file://` 安全错误

不要直接打开 HTML。按上面的“前端播放器启动”章节，用本地 HTTP 服务打开。
