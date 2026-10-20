import cv2
from ultralytics import YOLO
import time
import os

def run_inference(model_path, source_video, output_video):
    print(f"\n=============================================")
    print(f"--- 开始测试模型:{model_path} ---")

    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        return
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"【模型大小】: {file_size:.2f} MB")

    model = YOLO(model_path)

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频{source_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps , (width, height))

    total_boxes = 0
    total_time = 0.0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        results = model.predict(source=frame, conf=0.25, verbose=False)
        result = results[0]

        end_time = time.time()
        total_time += (end_time - start_time)
        frame_count += 1

        if result.boxes is not None:
            total_boxes += len(result.boxes)

        annotated_frame =result.plot()
        out.write(annotated_frame)

    cap.release()
    out.release()

    avg_time_ms =(total_time/frame_count) * 1000

    print(f"【总帧数】 : {frame_count}")
    print(f"【平均推理耗时】:{avg_time_ms:.2f} ms/帧")
    print(f"【检测目标总数】:{total_boxes}")
    print(f"【结果已保存至】:{output_video}")

if __name__ == "__main__":
    video_file = "armor_video.mp4"

    run_inference("best.pt",video_file,"output.mp4")

    if os.path.exists("best.torchscript"):
        run_inference("best.torchscript",video_file,"output_torchscript.mp4")
    else:
        print("\n未找到 best.torchscript，请先运行 export.py 进行导出！")