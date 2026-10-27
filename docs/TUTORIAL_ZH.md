# 使用教程

本项目计算我们已经定义的 **gait-conditioned relative roughness**：给定实际地形高度图，选定已保存的 wave number 和承腿节数，返回全连接足点坡角 RMS，单位为度。

## 1. 安装

推荐 Python 3.11。在终端运行：

```bash
git clone https://github.com/wxy108/Terrain-Roughness-Model.git
cd Terrain-Roughness-Model
python -m venv .venv
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

Windows/Linux 的 CPU 安装：

```bash
python -m pip install -r environment/requirements-cpu-tested.txt
python -m pip install -e . --no-deps
```

有兼容 NVIDIA GPU，使用本项目验证过的 CUDA 12.8 配置：

```bash
python -m pip install -r environment/requirements-cuda128-tested.txt
python -m pip install -e . --no-deps
```

最小运行依赖只有 NumPy 和 PyTorch。测试需要 pytest。`environment/full_package_inventory.*` 是原计算环境的完整包版本记录，包含其他研究工作使用的包，不必整份安装。无需安装 MuJoCo、IsaacLab 或连接原电脑的 X 盘。

## 2. 先运行已有实例

先用平地做快速安装检查，预期为 0 度：

```bash
terrain-roughness data/heightmaps/libo_x0p00.npz --wave 3 --device cpu
```

完整的非平地实例：

```bash
python examples/basic_usage.py
```

默认使用 Libo 地形高度 ×1、wave=3、19 个承腿节段。预期输出：

```text
R_g = 9.43447229 deg
```

或者：

```bash
terrain-roughness data/heightmaps/libo_x1p00.npz --wave 3 --segments 19
```

程序读取 NPZ 内的实际网格间距和坐标原点。默认优先用可用 GPU；加 `--device cpu` 可选择 CPU。

完整非平地扫描计算量很大。本机全新 CPU 环境的该实例在默认线程数和单线程设置下都超过了 180 秒的短测上限，因此未在该时限内验证其完整 CPU 输出。CPU 的解析测试和安装检查通过，全部 300 条件的完整复算使用 CUDA 完成。快速验证用上面的平地命令，完整扫描建议用 GPU。

## 3. 在 Python 中调用

```python
import numpy as np
from terrain_roughness import calculate_roughness

with np.load("data/heightmaps/libo_x1p00.npz") as terrain:
    R_deg = calculate_roughness(
        terrain["height"],
        wave_number=3.0,
        grid_spacing_m=float(terrain["dx_m"]),
        map_origin_xy_m=terrain["origin_xy_m"],
        segments=19,
    )

print(R_deg)
```

输入 `height_map` 是 `[ny, nx]` 的二维 NumPy 数组，高度单位为**米**。`grid_spacing_m` 是高度图相邻采样点在水平方向的实际间距，也用米；例如 0.5 mm 对应 `0.0005`。x/y 间距不同时可传 `(dx, dy)`。行向 +y 增加，列向 +x 增加。

`map_origin_xy_m` 指 `height_map[0, 0]` 的 xy 坐标，默认 `(0, 0)`。整体平移坐标不改变指标；选取不同大小的地形区域会改变被平均的地形。

## 4. 换成自己的地形

```python
height_map = np.load("my_height_map.npy")  # 先确保高度已经换成米
R_deg = calculate_roughness(
    height_map,
    wave_number=2.5,
    grid_spacing_m=0.0005,
    segments=15,
)
```

命令行对应：

```bash
terrain-roughness my_height_map.npy --wave 2.5 --segments 15 --spacing-mm 0.5
```

命令行中 `--spacing-mm`、`--origin-mm`、`--bounds-mm`、`--scan-step-mm` 用毫米；高度数组仍然用米。灰度 PNG 需要先明确灰度到实际高度的换算，不能直接当成米。

程序默认将整张高度图作为地形。若输入的是更大的 parent heightfield，可以用 `terrain_bounds_m=(xmin,xmax,ymin,ymax)` 指定真正参与扫描的地形范围。仓库中的 `data/heightmaps` 已经裁剪到之前实验的实际地形范围；`data/source` 保留未修改的大范围基准高度场。

## 5. wave number 和节数

支持的 wave number：`1.5, 2, 2.5, 3, 3.5, 4`。

支持的承腿节数：`11, 13, 15, 17, 19`，默认 `19`。

这些数值选择已保存的模型足点轨迹和 stance 状态，并包含之前确认过的各 wave 对应幅值。这里没有对其他 wave 做插值，也没有根据 wave number 临时猜测新的步态。

## 6. 一次计算做了什么

1. 一个周期取 50 个相位：0、0.02、…、0.98，不重复周期终点。
2. 每个相位取 36 个方向：0°、10°、…、350°，得到 1,800 个核。
3. 全部核在同一个共同 valid 区域平移，默认网格间距不超过 2 mm。
4. 每个位置只保留 stance 足点，读取地形高度，对所有无序足对计算 `atan(abs(Δh)/水平距离)`。
5. 对坡角平方，在足对、位置、相位和方向上平均，最后开方并转为度。

本数据的平移网格为 220×165，共 36,300 个位置。共同 valid 边界由全部 30 种组合的所有足点投影确定，包括 swing 足点；计算指标时仍只使用 stance 足点。

这个数值是几何上的 terrain challenge 指标。neutral configuration 定义的参考地形粗糙度是另一个固定窗口计算，旧图中的该参考值保存在 `results/reference_50x36.csv`。

## 7. 测试和复现

运行快速测试：

```bash
python -m pytest -q
```

运行三个完整实际地形回归测试：

```bash
python -m pytest -q --run-slow
```

重新计算完整 6 wave×5 节数×10 高度的 300 个结果，并和之前保存的数值逐项比较：

```bash
python studies/reproduce_study.py --device cuda --verify --output runs/reproduction
```

只复算 wave=3、19 节：

```bash
python studies/reproduce_study.py --device cuda --waves 3 --segments 19 --verify
```

加速版一次读取基准地形，通过精确的高度差缩放，同时计算九档非零高度。每档仍单独执行 `atan`；没有把最终 RMS 直接乘高度系数。

重新生成四张科学图：

```bash
python -m pip install -e ".[plots]"
python studies/plot_results.py --output runs/figures
```

## 8. GPU 编译

普通 GPU 计算不需要编译器。配置好 Torch/Triton 编译器后，可在 Python 传 `compile_gpu=True`，或在命令行加 `--compile-gpu`。首次运行会有编译时间。

本项目 Torch 2.7/CUDA 12.8 Windows 配置的可选依赖：

```bash
python -m pip install -r environment/requirements-compile-windows.txt
python -m pytest -q --run-compiled
```

GPU 计时对比：

```bash
python studies/benchmark.py --repeats 3 --output runs/benchmark.json
```

具体耗时取决于 GPU、地形范围、第一次编译和缓存。此前测得的 5.0×、7.6×加速及其全部计时记录也包含在仓库中。
