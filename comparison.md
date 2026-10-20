# 原始模型与 TorchScript 模型性能对比表

测试环境：Windows 11, CPU (Intel Core i9-14900HX)
测试视频：`armor_video.mp4` (共 634 帧)

| 对比项目 | 原始 .pt 模型 | 导出的 .torchscript 模型 |
| :--- | :--- | :--- |
| **模型格式** | PyTorch (.pt) | TorchScript (.torchscript) |
| **模型文件大小** | 5.62 MB | 9.94 MB |
| **平均推理耗时** | 45.97 ms/帧 | 41.39 ms/帧 |
| **检测目标总数** | 904 个 | 919 个 |
| **可视化结果** | 见 `output.mp4` | 见 `output_torchscript.mp4` |

## 分析结论
1. **大小**：TorchScript 包含了完整计算图，体积略大。
2. **速度**：TorchScript 经过静态图优化，推理速度提升约 10%。
3. **精度**：检测结果基本一致，微小差异属于正常的浮点精度误差。