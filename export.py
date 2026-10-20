from ultralytics import YOLO
import os

def main():
    model_path='best.pt'
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)

    print("开始导出 TorchScript 模型，请稍等...")
    torchscript_path =model.export(format='torchscript')

    print(f"导出成功!文件路径: {torchscript_path}")

    pt_size =os.path.getsize(model_path) / (1024 * 1024)
    ts_size =os.path.getsize(torchscript_path) / (1024 * 1024)
    print(f"\n【大小对比】")
    print(f"原始 .pt 模型大小: {pt_size:.2f} MB")
    print(f"导出 .torchscript 模型大小: {ts_size:.2f} MB")


if __name__ == '__main__':
    main()
