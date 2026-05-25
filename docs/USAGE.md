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

当前实时页面使用浏览器原生摄像头预览，不再通过 Python 重新编码视频。页面左上角可以选择摄像头和检测类别。

`Detection` 开关默认关闭。关闭时只播放摄像头；打开后，浏览器会从当前 cover 视口里截取实际可见区域，缩放后发送给 Python 检测，再把返回的框按视口坐标叠加到本地视频上。

这个架构的目标是让视频播放和 YOLO 检测解耦：视频由浏览器直接播放，检测慢时只影响框更新频率，不应该拖慢视频播放。

页面右上角：

```text
Size    摄像头实际采集分辨率
FPS     检测请求返回频率，不是视频播放帧率
Detect  单次后端检测耗时
```

常用启动方式：

```powershell
python scripts\live_track_server.py
python scripts\live_track_server.py --device 0
python scripts\live_track_server.py --model yolo26n.pt --device 0 --imgsz 480
```

检测压力主要由两个参数控制：

```text
--detect-width  浏览器发给后端的检测图宽度，默认 480
--imgsz         YOLO 推理尺寸，越小越快
```

如果框更新明显滞后，优先降低检测图宽度和推理尺寸：

```powershell
python scripts\live_track_server.py --detect-width 416 --imgsz 416
```

如果要更准一些：

```powershell
python scripts\live_track_server.py --detect-width 640 --imgsz 640
```

实时接口：

```text
/             实时预览页面
/classes      当前类别和可选类别
/set-classes  更新检测类别
/detect-frame 浏览器抽帧检测接口
/config       前端读取检测参数
```

性能建议：

1. NVIDIA 显卡优先使用 `--device 0`。
2. 先用 `yolo11n.pt` 或 `yolo26n.pt`，稳定后再换 `s / m`。
3. 在页面里只选择需要的类别。
4. 用 `--detect-width` 和 `--imgsz` 控制检测速度和精度。
5. 页面右上角 `Detect` 如果经常超过 200ms，优先降低 `--detect-width` 或 `--imgsz`。

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
