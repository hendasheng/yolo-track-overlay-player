# 使用文档

本文档记录项目的详细使用方式，可作为 GitHub Wiki 页面源稿。

## 环境激活

Windows CPU：

```powershell
.\yolo-cpu-env\Scripts\Activate.ps1
```

Windows GPU：

```powershell
conda activate .\yolo-gpu-env
```

macOS：

```bash
source yolo-cpu-env/bin/activate
```

## 离线视频检测

默认运行：

```powershell
python scripts\track_objects_stickers.py
```

默认行为：

- 读取 `source/` 中最后修改时间最新的 `.mp4`
- 使用 `yolo11n.pt`
- 检测 `car`
- 使用 `conf=0.15`
- 输出到 `runs/`

指定视频：

```powershell
python scripts\track_objects_stickers.py --source source\your_video.mp4
```

macOS：

```bash
python scripts/track_objects_stickers.py --source source/your_video.mp4
```

使用 GPU：

```powershell
python scripts\track_objects_stickers.py --device 0
```

## 常用检测命令

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

## 支持的类别

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

## 模型选择

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

## 实时摄像头检测

启动实时服务：

```powershell
python scripts\live_track_server.py
```

打开：

```text
http://127.0.0.1:8765
```

页面左上角会显示可打开的摄像头，点击即可切换；检测类别也在页面里切换。Windows 会优先显示 DirectShow 设备名，macOS 会优先显示 AVFoundation 设备名；没有 FFmpeg 时会退回数字 source。

常用启动方式：

```powershell
python scripts\live_track_server.py
python scripts\live_track_server.py --device 0
python scripts\live_track_server.py --model yolo26n.pt --device 0 --imgsz 640
```

实时页面会用 cover 样式铺满窗口，检测框会按同样的裁切和缩放映射。拖拽改变窗口尺寸时，视频和 overlay 会跟着重新适配。页面右上角的 `Size` 会显示当前实际采集分辨率。

默认会用 `--resolution-mode balanced`，目标是 720p 左右的像素量，避免在低清和超高清卡顿之间来回跳。这里不是强行限制 16:9 比例；脚本会尝试 16:9、4:3 等常见档位，并按摄像头实际返回的宽高显示和映射检测框。MJPEG 会用较高 JPEG 质量输出到浏览器。

如果要更流畅：

```powershell
python scripts\live_track_server.py --resolution-mode speed
```

如果要更清晰，但最高只自动尝试到 1080p：

```powershell
python scripts\live_track_server.py --resolution-mode quality
```

如果要固定请求 1080p：

```powershell
python scripts\live_track_server.py --width 1920 --height 1080 --jpeg-quality 95
```

如果要完全保留摄像头默认输出，不做自动高清探测：

```powershell
python scripts\live_track_server.py --no-auto-resolution
```

如果画面变流畅但推理跟不上，可以保持摄像头 1080p，同时用较小推理尺寸：

```powershell
python scripts\live_track_server.py --width 1920 --height 1080 --jpeg-quality 95 --imgsz 640
```

网络摄像头或视频流：

```powershell
python scripts\live_track_server.py --source rtsp://user:pass@camera-ip/stream --device 0
```

实时接口：

```text
/             实时预览页面
/video        MJPEG 视频流
/events       Server-Sent Events，每帧一条 JSON
/latest.json  最新一帧 JSON
/classes      当前类别和可选类别
/set-classes  更新检测类别
```

性能建议：

1. NVIDIA 显卡优先使用 `--device 0`。
2. 先用 `yolo11n.pt` 或 `yolo26n.pt`，稳定后再换 `s / m`。
3. 在页面里只选择需要的类别。
4. 用 `--imgsz 512` 或 `--imgsz 640` 控制速度和精度。
5. 推理跟不上摄像头时，用 `--max-fps 15` 或 `--max-fps 20` 限制处理帧率。

摄像头排查：

```powershell
python scripts\live_track_server.py --probe-cameras 6 --backend dshow
python scripts\live_track_server.py --probe-cameras 6 --backend msmf
```

macOS：

```bash
python scripts/live_track_server.py --probe-cameras 6 --backend avfoundation
```

## 输出结果

每次离线运行都会在 `runs/` 下新建文件夹。

例子：

```text
runs/yolo11m_person_gpu_conf0p15_20260521_180000/
  your_video_stickers.mp4
  your_video_tracks.csv
  your_video_tracks.jsonl
```

文件说明：

```text
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

Windows：

```powershell
cd web_track_overlay_player
..\yolo-cpu-env\Scripts\python.exe -m http.server 8090 --bind 127.0.0.1
```

macOS：

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
