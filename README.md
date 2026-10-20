# YOLO 模型导出与推理性能对比

## 1. 项目简介
本项目为计算机视觉课程第二次作业。在上一次 YOLO 模型推理作业的基础上，本项目重点学习了如何将 PyTorch 格式的模型（`.pt`）导出为其他格式（本次选用 **TorchScript** 格式），并使用导出后的模型重新进行视频推理。

通过本项目，成功实现了对原始视频的检测框、类别、置信度以及关键点的可视化，并系统地对比了原始模型与导出模型在文件大小、推理耗时和检测效果上的差异。

## 2. 环境依赖
*   **操作系统**：Windows 11
*   **Python 版本**：3.10.11
*   **核心依赖库**：
    *   `ultralytics` (用于模型加载、导出与推理)
    *   `opencv-python` (用于视频处理)
    *   `torch` (PyTorch 底层支持)

## 3. 模型导出方法
运行 `export.py` 脚本即可将原始 `best.pt` 导出为 `best.torchscript`。
导出核心代码：
```python
from ultralytics import YOLO
model = YOLO('best.pt')
model.export(format='torchscript')
```

(注：在最新版本的 Ultralytics 中，导出 torchscript 已不再支持 optimize=True 参数。)

4. 运行与推理说明

运行 inference.py 脚本，该脚本会自动加载原始模型和导出模型，对同一个测试视频进行推理，并记录耗时和检测目标数量。

运行命令：

```powershell
python inference.py
```

5. 实验结果对比

以下是使用相同测试视频（共 634 帧，ARMOR 装甲板检测）的性能对比数据：

对比项目 原始 .pt 模型 导出的 .torchscript 模型
模型格式 PyTorch 动态图 TorchScript 静态图
模型文件大小 5.62 MB 9.94 MB
平均推理耗时 45.97 ms/帧 41.39 ms/帧
检测目标总数 904 个 919 个
检测效果保存 output.mp4 output_torchscript.mp4

6. 实验结论与问题分析

· 关于文件大小：导出的 TorchScript 模型（9.94 MB）比原始 .pt（5.62 MB）大。这是因为 TorchScript 为了脱离 Python 环境独立运行，打包了完整的计算图和网络结构。
· 关于推理速度：TorchScript 的推理速度（41.39 ms/帧）比原版（45.97 ms/帧）快了约 10%。这是因为静态图在底层进行了算子融合，减少了 Python 解释器的开销。
· 关于检测差异：TorchScript 检测出 919 个目标，比原模型多 15 个。这主要是由于底层浮点数计算的精度误差，导致个别置信度在 0.25 边缘的框被保留。整体检测效果肉眼观察一致。
· 遇到的问题：在导出时尝试使用 optimize=True 报错，查询后发现新版 Ultralytics 已废弃该参数，移除后解决。推理时出现 torch.jit.load is deprecated 警告，属于 PyTorch 版本更迭的正常提示，不影响运行。
