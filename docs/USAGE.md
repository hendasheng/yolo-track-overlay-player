# 使用文档

本文档记录项目的详细使用方式，可作为 GitHub Wiki 页面源稿。

实时摄像头检测已迁移到独立的 `rfdetr-live` 项目。当前仓库保留离线视频检测、跟踪数据导出与前端叠加播放器。

## 环境激活

Windows CPU：

```powershell
.\video-track-cpu-env\Scripts\Activate.ps1
```

退出：

```powershell
deactivate
```

Windows GPU：

```powershell
conda activate .\video-track-gpu-env
```

退出：

```powershell
conda deactivate
```

macOS：

```bash
source video-track-cpu-env/bin/activate
```

退出：

```bash
deactivate
```

## RF-DETR 离线视频检测

推荐用于规避 YOLO / Ultralytics 许可证约束的离线检测流程。默认使用 RF-DETR Medium，输出格式和 YOLO 脚本一致，可以继续用同一个前端叠加播放器。

安装依赖：

```powershell
pip install -r requirements-rfdetr.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

检测车辆：

```powershell
python scripts\track_objects_rfdetr.py --classes car
```

检测人：

```powershell
python scripts\track_objects_rfdetr.py --classes person
```

RF-DETR 预训练 COCO 模型没有独立的 `face` / 面部类别。`--classes person` 只能检测整个人体或人像区域，不能得到面部框。需要检测面部时，使用 YOLO 人脸权重：

```powershell
python scripts\track_objects_yolo.py --model yolov11n-face.pt --classes 0
```

分割人像：

```powershell
python scripts\track_objects_rfdetr.py --model-size seg-small --classes person --device cuda
```

同时画分割区域和检测框：

```powershell
python scripts\track_objects_rfdetr.py --model-size seg-small --classes person --device cuda --render both
python scripts\track_objects_rfdetr.py --model-size seg-small --classes person --device cuda --mask-alpha 0.5
```

RF-DETR 预训练 COCO 模型使用 COCO 原始 category ID，不是 YOLO 的 0-79 连续编号。常用类别：

```text
person  1
bicycle 2
car     3
bus     6
truck   8
bird    16
cat     17
dog     18
```

如果想先看模型原始输出，不做类别过滤，可以用下面的诊断命令。它只跑前 5 帧，不能作为正式处理命令：

```powershell
python scripts\track_objects_rfdetr.py --classes all --device cuda --conf 0.2 --debug-detections 5 --max-frames 5
```

正式处理完整视频时不要带 `--debug-detections` 和 `--max-frames`：

```powershell
python scripts\track_objects_rfdetr.py --classes person --device cuda --conf 0.2
```

RF-DETR 脚本默认每帧刷新进度。如果想减少终端输出：

```powershell
python scripts\track_objects_rfdetr.py --classes person --device cuda --progress-every 10
```

在交互终端中，进度会在同一行动态刷新；如果输出被 IDE 或日志系统捕获，则按 `--progress-every` 间隔逐行输出。

指定 GPU：

```powershell
python scripts\track_objects_rfdetr.py --classes person --device cuda
```

强制 CPU：

```powershell
python scripts\track_objects_rfdetr.py --classes person --device cpu
```

Apple Silicon Mac 可以尝试 MPS：

```bash
python scripts/track_objects_rfdetr.py --classes person --device mps
```

指定模型大小：

```powershell
python scripts\track_objects_rfdetr.py --model-size small --classes car
python scripts\track_objects_rfdetr.py --model-size large --classes person --conf 0.4
```

如果检测太少，降低置信度；如果重叠框太多，提高置信度或调低 NMS 阈值：

```powershell
python scripts\track_objects_rfdetr.py --classes person --device cuda --conf 0.2 --nms-threshold 0.45
```

默认会调用 RF-DETR 的 `optimize_for_inference()`。如果遇到兼容性问题，可临时关闭：

```powershell
python scripts\track_objects_rfdetr.py --classes person --device cuda --no-optimize
```

可选模型大小：

```text
nano
small
medium
large
seg-nano
seg-small
seg-medium
seg-large
```

这些默认档位对应 RF-DETR 的 Apache 2.0 开源模型线。不要把 `plus` / XL / 2XLarge 模型混进当前脚本，除非你明确接受对应的额外许可条款。

普通 RF-DETR 档位输出检测框；`seg-*` 档位输出检测框和实例分割 mask。默认 `--render auto` 会在 mask 可用时画分割区域，`mask_polygon` 字段也会写入分割轮廓。

## YOLO 离线视频检测

默认运行：

```powershell
python scripts\track_objects_yolo.py
```

默认行为：

- 读取 `source/` 中最后修改时间最新的 `.mp4`
- 使用 `yolo11n.pt`
- 检测 `car`
- 使用 `conf=0.15`
- 输出到 `runs/`

指定视频：

```powershell
python scripts\track_objects_yolo.py --source source\your_video.mp4
```

macOS：

```bash
python scripts/track_objects_yolo.py --source source/your_video.mp4
```

使用 GPU：

```powershell
python scripts\track_objects_yolo.py --device 0
```

## 常用检测命令

检测车：

```powershell
python scripts\track_objects_yolo.py --classes car
```

检测人：

```powershell
python scripts\track_objects_yolo.py --classes person
```

检测面部：

```powershell
python scripts\track_objects_yolo.py --model yolov11n-face.pt --classes 0
```

`yolov11n-face.pt` 是为面部检测下载的 YOLO 人脸模型。它不是 COCO 通用模型，类别不要写成 `person`；通常使用 `--classes 0` 过滤 face 类。需要更高精度时可以换成 `yolov11l-face.pt`。

同时检测人和车：

```powershell
python scripts\track_objects_yolo.py --classes person,car
```

检测鸟：

```powershell
python scripts\track_objects_yolo.py --classes bird
```

降低置信度，减少漏检：

```powershell
python scripts\track_objects_yolo.py --classes person --conf 0.1
```

使用更大的模型：

```powershell
python scripts\track_objects_yolo.py --model yolo11l.pt --classes person --device 0
```

每一帧都显示进度：

```powershell
python scripts\track_objects_yolo.py --progress-every 1
```

在交互终端中，进度会在同一行动态刷新；如果输出被 IDE 或日志系统捕获，则按 `--progress-every` 间隔逐行输出。

关闭进度显示：

```powershell
python scripts\track_objects_yolo.py --progress-every 0
```

只跑前几帧做验证：

```powershell
python scripts\track_objects_yolo.py --classes person --device 0 --max-frames 3 --progress-every 1
```

## 离线视频分割

使用 `*-seg.pt` 模型即可输出分割区域；默认 `--render auto` 会自动选择框或分割。

```powershell
python scripts\track_objects_yolo.py --model yolo11n-seg.pt --classes person
```

常用覆盖项：

```powershell
python scripts\track_objects_yolo.py --model yolo11n-seg.pt --classes person --render both
python scripts\track_objects_yolo.py --model yolo11n-seg.pt --classes person --mask-alpha 0.5
```

```text
--render auto  默认。普通模型画框，分割模型画分割区域
--render box   强制只画框
--render mask  强制只画分割区域
--render both  同时画分割区域、检测框和 ID 贴纸
```

## 支持的类别

脚本内置了一些常用别名：

```text
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, bird, cat, dog, horse, sheep, cow
```

也可以直接用 COCO 类别 ID：

```powershell
python scripts\track_objects_yolo.py --classes 0,2
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

## 模型选择

```text
yolo11n.pt  最快，容易漏检
yolo11s.pt  小模型，比 n 稳
yolo11m.pt  中模型，精度和速度折中
yolo11l.pt  大模型，更准更慢
yolo11x.pt  最大，最慢
```

分割模型对应使用：

```text
yolo11n-seg.pt
yolo11s-seg.pt
yolo11m-seg.pt
yolo11l-seg.pt
yolo11x-seg.pt
```

面部检测使用专门的人脸权重：

```text
yolov11n-face.pt  面部检测小模型，速度快
yolov11l-face.pt  面部检测大模型，更准更慢
```

运行人脸模型时通常配合 `--classes 0`，不要使用 COCO 的 `person` 类别名。

建议顺序：

```text
n -> s -> m -> l -> x
```

CPU 环境建议用 `n / s`。GPU 环境可以尝试 `m / l / x`。

## 输出结果

每次离线运行都会在 `runs/` 下新建文件夹。

例子：

```text
runs/yolo11m_person_gpu_conf0p15_20260521_180000/
  your_video.mp4
  your_video_stickers.mp4
  your_video_tracks.csv
  your_video_tracks.jsonl
```

文件说明：

```text
your_video.mp4    原始视频副本，文件名与输入文件一致
*_stickers.mp4   带检测框和贴纸的视频
*_tracks.csv     表格数据，适合 Excel / WPS 查看
*_tracks.jsonl   逐帧 JSON 数据，适合程序读取
```

## CSV / JSONL 字段

`tracks.csv` 每一行是一帧里的一个目标。

主要字段：

```text
frame             第几帧
track_id          跟踪 ID，同一个目标会尽量保持同一个 ID
class_id          COCO 类别 ID
class_name        类别名
confidence        置信度
color             跟踪颜色，Web 叠加播放器会优先使用
x1, y1, x2, y2    检测框坐标，单位是原视频像素
center_x/y        中心点坐标，单位是原视频像素
center_x/y_norm   归一化中心点，范围 0-1
mask_polygon      分割轮廓点。只有使用分割模型并生成 mask 时才有值
```

`tracks.jsonl` 是一行一帧：

```json
{"frame": 1, "objects": [...]}
{"frame": 2, "objects": [...]}
{"frame": 3, "objects": [...]}
```

## 前端叠加播放器

前端页面在：

```text
web_track_overlay_player/
```

不要直接双击 `index.html`。浏览器的 `file://` 安全限制可能导致拖拽视频失败。

不论 CPU 还是 GPU 环境，任何一个可用的 Python 都能启动，选当前已有的即可。

Windows（CPU venv 环境）：

```powershell
cd web_track_overlay_player
..\video-track-cpu-env\Scripts\python.exe -m http.server 8090 --bind 127.0.0.1
```

Windows（GPU conda 环境）：

```powershell
cd web_track_overlay_player
..\video-track-gpu-env\python.exe -m http.server 8090 --bind 127.0.0.1
```

macOS：

```bash
cd web_track_overlay_player
../video-track-cpu-env/bin/python -m http.server 8090 --bind 127.0.0.1
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
3. 确认类别选择正确
4. 使用 GPU 后再尝试更大模型

拖拽视频到前端时报 `file://` 安全错误

不要直接打开 HTML。按“前端叠加播放器”章节，用本地 HTTP 服务打开。
