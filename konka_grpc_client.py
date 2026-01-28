import datetime
import glob
import json
import os
import pickle
import queue
import threading
import time

import cv2
import grpc
import numpy as np
import yaml
from sklearn.metrics.pairwise import cosine_similarity

import imgs_pb2
import imgs_pb2_grpc
from utils.logger import SingletonLogger

# 配置日志系统
logger = SingletonLogger(
    name='pipeline',
    log_file='logs/pipeline.log',
    level='INFO',
    console=True
)


def get_color(track_id):
    np.random.seed(int(track_id))
    return tuple(np.random.randint(0, 255, 3).tolist())


def load_vocab(path="en_vocabulary.txt"):
    with open(path, 'r') as f:
        vocab = [line.strip() for line in f.readlines()]
    word2index = {word: idx for idx, word in enumerate(vocab)}
    logger.info("vocab loaded")
    return vocab, word2index


class KonkaGrpcClient:
    def __init__(self, frame_queue=None, result_queue=None):
        default_config_path = "client_config.yaml"
        config = {}

        # 尝试加载 YAML 配置文件
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {default_config_path}")
        else:
            logger.warning(f"Config file not found at {default_config_path}, using defaults...")

        self.cfg = config

        # 网络配置
        self.server_ip = config.get('server_ip', '0.0.0.0')
        self.server_port = config.get('server_port', '50051')
        self.data_size = config.get('data_size', 500)
        self.draw_result = config.get('draw_result', True)
        self.result_queue = result_queue

        # 输入源配置
        input_cfg = config.get('input', {})
        self.input_type = input_cfg.get('type', 'camera')
        self.input_source = input_cfg.get('source', '0')
        self.frame_queue = frame_queue

        # 初始化图像输入相关变量
        self.cap = None  # VideoCapture 对象
        self.image_files = None  # 图像路径列表
        self.current_image_index = 0  # 图像索引计数器

        self.vocab, self.word2index = load_vocab()
        self.send_masks = config.get('send_masks', True)

        self.width = config.get("width")
        self.height = config.get("height")

        self.filter_tracks = config.get('filter_tracks')

        self.padding = int(max(self.width, self.height) * 0.4)
        self.extended_width = self.width + 2 * self.padding
        self.extended_height = self.height + 2 * self.padding

    def image_generator(self):
        frame_counter = 0
        send_masks = self.send_masks

        while True:
            # 如果输入类型是队列，则从队列中获取帧
            if self.input_type == 'queue':
                if self.frame_queue is None:
                    logger.error("Frame queue is None.")
                    break
                try:
                    queue_item = self.frame_queue.get(timeout=5)  # 设置超时以避免无限阻塞

                    # 处理不同的队列项格式
                    if isinstance(queue_item, tuple) and len(queue_item) == 2:
                        # 来自图片文件夹的帧，包含元数据
                        frame, timestamp_str = queue_item
                        # 使用文件名作为时间戳
                        time_stamp = timestamp_str
                    else:
                        # 来自视频的帧，保持原有逻辑
                        frame = queue_item
                        time_stamp = time.time() * 1000

                except queue.Empty:
                    logger.warning("Frame queue is empty.")
                    continue
            else:
                # 如果不是队列模式，则从摄像头或视频文件读取帧
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame from video source.")
                    break
                time_stamp = time.time() * 1000

            # 统一时间戳格式，确保键的一致性
            if isinstance(time_stamp, (int, float)):
                time_stamp_key = str(time_stamp)  # 用作字典键的时间戳
                time_bytes = str(time_stamp).encode("utf-8")  # 发送的时间戳
            else:
                time_stamp_key = time_stamp  # 文件名等字符串直接用作键
                time_bytes = time_stamp.encode("utf-8")  # 发送的时间戳

            img_bytes = frame.tobytes()

            if not img_bytes or not time_bytes:
                logger.warning("Empty fields, skipping.")
                continue

            try:
                proto_msg = imgs_pb2.Img(img=img_bytes, timeStamp=time_bytes, requiresMasks=send_masks)
                yield proto_msg, time_stamp_key, frame  # 使用统一的键
                frame_counter += 1
            except Exception as e:
                logger.error(f"Failed to create Img proto: {e}")
                break

    def request_iterator(self, sent_frames):
        for proto, timestamp, frame in self.image_generator():
            if frame is not None:
                sent_frames[timestamp] = frame.copy()
            yield proto

    def run(self):
        sent_frames = {}

        # 初始化图像源
        self.cap = None
        self.image_files = None
        self.current_image_index = 0

        if self.input_type == 'queue':
            # 使用外部队列（需上游注入 frame_queue）
            if self.frame_queue is None:
                logger.error("External input mode requires a valid frame_queue.")
                return False
            frame = self.frame_queue.get()
        else:
            # 默认处理摄像头或视频文件（camera / video）
            try:
                source = int(self.input_source)
            except ValueError:
                source = self.input_source

            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                logger.error(f"Error: Could not open video source: {source}")
                return False
            self.image_files = None
            ret, frame = self.cap.read()

        # 初始化 VideoWriter（仅当需要绘制时）
        writer = None
        output_path = self.cfg.get('output_path')
        if self.draw_result:
            fps = 10
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (self.extended_width, self.extended_height))
            if not writer.isOpened():
                logger.warning("Error: Could not open video writer.")
                return False

        with grpc.insecure_channel(
                f"{self.server_ip}:{self.server_port}",
                options=[
                    ('grpc.max_send_message_length', self.data_size * 1024 * 1024),
                    ('grpc.max_receive_message_length', self.data_size * 1024 * 1024)
                ]
        ) as channel:

            logger.info(f"connected to {self.server_ip}:{self.server_port}")

            stub = imgs_pb2_grpc.ImageProcessingStub(channel)
            logger.info("stub created")

            responses = stub.continuousInfo(self.request_iterator(sent_frames))

            frame_times = []  # 存储最近 N 帧的时间，用于滑动平均 FPS
            window_size = 100
            first_request_time = None
            last_response_time = None
            frame_count = 0

            # 在处理响应的循环中
            for response in responses:
                if response is None:
                    continue

                receive_time = time.perf_counter()
                frame_count += 1

                try:
                    # 解析时间戳并保持类型一致性
                    send_time_str = response.timeStamp.decode("utf-8")
                    send_time = send_time_str  # 使用字符串作为键
                except Exception as e:
                    logger.error(f"Failed to decode timestamp: {e}")
                    continue

                # 记录第一个请求的发送时间
                if first_request_time is None:
                    first_request_time = float(send_time_str) if send_time_str.replace('.', '').isdigit() else 0

                # 更新最后一个响应的接收时间
                last_response_time = receive_time

                latency = receive_time - (
                    float(send_time_str) if send_time_str.replace('.', '').isdigit() else receive_time)
                fps = 1.0 / latency if latency > 0 else float('inf')

                frame_times.append(latency)
                if len(frame_times) > window_size:
                    frame_times.pop(0)

                avg_latency = sum(frame_times) / len(frame_times)
                avg_fps = 1.0 / avg_latency if avg_latency > 0 else float('inf')

                # 使用一致的键类型获取图像
                img = sent_frames.pop(send_time).copy()  # 使用字符串键

                tracks = []
                if len(response.tracks) > 0:
                    for track in response.tracks:
                        mask_coords = []
                        for mask_row in track.mask:
                            x, y = mask_row.x, mask_row.y
                            mask_coords.append([x, y])
                        tracks.append(
                            [track.x1, track.y1, track.x2, track.y2, track.trackId, track.conf,
                             track.cls, mask_coords, track.isMove, track.feats])

                if self.filter_tracks:
                    tracks = filter_embed_tracks(
                        tracks=tracks,
                        whitelist_file="en_blacklist_embed.json",
                        blacklist_file="en_whitelist_embed.json",
                        enable_whitelist=False,
                        enable_blacklist=False,
                        whitelist_conf_threshold=0.5,
                        blacklist_conf_threshold=0.4
                    )

                self.result_queue.put_nowait((tracks, send_time))

                if self.draw_result:
                    # 计算带扩展区域的画布大小
                    padding = self.padding
                    new_width = img.shape[1] + 2 * padding
                    new_height = img.shape[0] + 2 * padding

                    # 创建新画布并把原图放在中间
                    extended_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                    extended_img[padding:padding + img.shape[0], padding:padding + img.shape[1]] = img
                    if len(tracks) > 0:
                        for track in tracks:
                            x1, y1, x2, y2 = map(round, track[:4])
                            track_id = track[4]
                            print(f"+++ track_id: {track_id} +++")
                            conf = track[5]
                            cls = track[6]
                            cls_id = self.word2index[cls]
                            print(f"+++ cls: {cls} +++")
                            color = get_color(cls_id)

                            # 调整坐标以适配新画布
                            x1 += padding
                            y1 += padding
                            x2 += padding
                            y2 += padding

                            label = f"{cls}_{conf:.2f}_{track_id}"
                            cv2.rectangle(extended_img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(extended_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                            # 绘制掩码
                            contour = np.array(track[7]).reshape(-1, 1, 2).astype(int)

                            # 调整mask坐标
                            contour[:, :, 0] += padding  # x 坐标偏移
                            contour[:, :, 1] += padding  # y 坐标偏移

                            mask = np.zeros(extended_img.shape[:2], dtype=np.uint8)
                            cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
                            mask_ = (mask > 0.5).astype(np.uint8)
                            extended_img[mask_ == 1] = extended_img[mask_ == 1] * 0.5 + np.array(color) * 0.5

                    cv2.imshow("Result", extended_img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
                    writer.write(extended_img)

            if first_request_time is not None and last_response_time is not None:
                total_duration = last_response_time - first_request_time
                logger.info(f"\nTotal processing time for {frame_count} frames: {total_duration:.2f} seconds")
                if frame_count > 0:
                    overall_fps = frame_count / total_duration
                    logger.info(f"Overall FPS: {overall_fps:.2f}")


def filter_embed_tracks(tracks, whitelist_file, blacklist_file, enable_whitelist, enable_blacklist,
                        whitelist_conf_threshold, blacklist_conf_threshold):
    if enable_whitelist:
        with open(whitelist_file, 'r') as f:
            whitelist = json.load(f)
            tracks_new = []
            for track in tracks:
                feat = track[9]
                for name, embed in whitelist.items():
                    similarity = cosine_similarity(feat, embed)[0][0]
                    if similarity > whitelist_conf_threshold:
                        tracks_new.append(track)
                        continue
        return tracks_new

    if enable_blacklist:
        with open(blacklist_file, 'r') as f:
            blacklist = json.load(f)
            for track in tracks:
                feat = track[9]
                for name, embed in blacklist.items():
                    similarity = cosine_similarity(feat, embed)[0][0]
                    if similarity > blacklist_conf_threshold:
                        tracks.remove(track)
                        break
        return tracks

def save_embeddings(embeddings, filename):
    """
    将图像特征字典保存到本地文件

    Args:
        embeddings (dict): 图像特征字典 {标签: embed}
        filename (str): 保存的文件名
    """
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)


def load_embeddings(filename):
    """
    从本地文件加载图像特征字典

    Args:
        filename (str): 要加载的文件名

    Returns:
        dict: 图像特征字典 {标签: embed}
    """
    with open(filename, 'rb') as f:
        embeddings = pickle.load(f)

    return embeddings


def load_media_to_queue(media_path, frame_queue, delay=0.05):
    """
    加载媒体文件（视频或图片文件夹）到队列中

    Args:
        media_path (str): 视频文件路径或图片文件夹路径
        frame_queue (queue.Queue): 帧队列
        delay (float): 每帧之间的时间延迟，默认0.05秒
    """
    # 检查路径是文件还是目录
    if os.path.isfile(media_path):
        # 处理视频文件
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {media_path}")
            return

        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("视频读取完成")
                break

            # 将帧放入队列
            frame_queue.put(frame)
            time.sleep(delay)  # 控制帧率（可选）

            i += 1

        cap.release()

    elif os.path.isdir(media_path):
        # 处理图片文件夹
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []

        # 获取所有图片文件
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(media_path, extension)))
            image_files.extend(glob.glob(os.path.join(media_path, extension.upper())))

        # 按文件名排序
        image_files.sort()

        if not image_files:
            print(f"在文件夹 {media_path} 中未找到图片文件")
            return

        print(f"找到 {len(image_files)} 个图片文件")

        for image_path in image_files:
            # 读取图片
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"无法读取图片文件: {image_path}")
                continue
            # if frame.shape[0] != 480 or frame.shape[1] != 640:
            #     frame = cv2.resize(frame, (480, 640))

            # 将帧放入队列
            try:
                frame_queue.put(frame, timeout=1)
            except queue.Full:
                print("队列已满，跳过一帧")
                continue

            time.sleep(delay)  # 控制帧率（可选）

        print("所有图片处理完成")

    else:
        print(f"路径既不是文件也不是目录: {media_path}")


def load_images_to_queue(image_folder, frame_queue, delay=0.05):
    """
    从图片文件夹加载图片到队列中，使用文件名作为时间戳标识符

    Args:
        image_folder (str): 图片文件夹路径
        frame_queue (queue.Queue): 帧队列
        delay (float): 每帧之间的时间延迟，默认0.05秒
    """
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []

    # 获取所有图片文件
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, extension)))
        image_files.extend(glob.glob(os.path.join(image_folder, extension.upper())))

    # 按文件名排序
    image_files.sort()

    if not image_files:
        print(f"在文件夹 {image_folder} 中未找到图片文件")
        return

    print(f"找到 {len(image_files)} 个图片文件")

    for image_path in image_files:
        # 读取图片
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片文件: {image_path}")
            continue

        # 调整图片尺寸为 (w=672, h=378)
        # frame = cv2.resize(frame, (378, 672))

        # 使用文件名（不含扩展名）作为时间戳
        filename = os.path.splitext(os.path.basename(image_path))[0]
        # 将文件名放入帧的元数据中
        frame_with_metadata = (frame, filename)

        # 将帧放入队列
        try:
            frame_queue.put(frame_with_metadata, timeout=1)
        except queue.Full:
            print("队列已满，跳过一帧")
            continue

        time.sleep(delay)  # 控制帧率（可选）

    print("所有图片处理完成")


def append_tracking_result_scannet(tracks, timestamp, base_url):
    """
    将单次跟踪结果追加到结果数组中

    Args:
        tracks: 跟踪结果列表
        timestamp: 时间戳/图片文件名
        base_url: 数据集基础路径
    """
    # 构建文件路径
    image_rgb_url = f"{base_url}/color_re/{timestamp}.jpg"
    image_depth_url = f"{base_url}/depth/{timestamp}.png"

    # 读取world2camera矩阵
    with open(f"{base_url}/pose/{timestamp}.txt", "r") as f:
        lines = f.readlines()
        world2camera = np.array([[float(x) for x in line.split()] for line in lines]).tolist()

    # 处理每个跟踪对象
    instance = []
    for track in tracks:
        x1, y1, x2, y2 = map(round, track[:4])
        track_id = track[4]
        cls = track[6]
        segmentation = np.array(track[7]).reshape(-1, 2).astype(int).tolist()
        features = np.array(track[10]).astype(np.float32).tolist()
        name = f"{cls}_{track_id}"

        obj = {
            "detection_2d": {"left_top": [x1, y1], "right_bottom": [x2, y2]},
            "object_id": str(track_id),
            "segmentation": segmentation,
            "features": features,
            "name": name
        }
        instance.append(obj)

    # 组装单次结果数据
    single_result = {
        "image_rgb_url": image_rgb_url,
        "image_depth_url": image_depth_url,
        "world2camera": world2camera,
        "instance": instance,
        "timestamp": timestamp,
    }

    # 追加到全局数组
    tracking_results.append(single_result)
    print(f"已添加第 {len(tracking_results)} 条记录，时间戳: {timestamp}")


def append_tracking_result_vggt(tracks, timestamp, base_url, json_file_path):
    """
    向已有的JSON文件中添加每个图像的instance内容

    Args:
        tracks: 跟踪结果列表
        timestamp: 时间戳/图片文件名
        base_url: 数据集基础路径
        json_file_path: JSON文件路径
    """
    image_rgb_url = f"{base_url}/rgb/{timestamp}"
    # 处理每个跟踪对象生成instance
    instance = []
    for track in tracks:
        x1, y1, x2, y2 = map(round, track[:4])
        track_id = track[4]
        cls = track[6]
        segmentation = np.array(track[7]).reshape(-1, 2).astype(int).tolist()
        features = np.array(track[9]).astype(np.float32).tolist() if len(track) > 9 and track[9] else []
        name = f"{cls}_{track_id}"

        obj = {
            "detection_2d": {"left_top": [x1, y1], "right_bottom": [x2, y2]},
            "object_id": str(track_id),
            "segmentation": segmentation,
            "features": features,
            "name": name
        }
        instance.append(obj)

    # 加载现有JSON数据
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                print(f"错误: 无法解析JSON文件 {json_file_path}")
                return
    else:
        logger.error(f"错误: JSON文件 {json_file_path} 不存在")
        return

    # 查找匹配timestamp的记录并更新instance字段
    updated = False
    for item in existing_data:
        if os.path.splitext(item.get("image_rgb_url"))[0] == image_rgb_url:
            # 更新现有记录的instance字段
            item["instance"] = instance
            updated = True
            break

    if not updated:
        logger.warning(f"警告: 未找到timestamp为 {timestamp} 的记录")

    # 保存更新后的数据
    with open(json_file_path, 'w') as f:
        json.dump(existing_data, f, indent=2)

    logger.info(f"已为timestamp {timestamp} 添加了 {len(instance)} 个实例")


if __name__ == '__main__':
    base_url = "/home/ubuntu/Downloads/rgb_part_depth"
    tracking_results = []
    frame_queue = queue.Queue(maxsize=100)
    result_queue = queue.Queue(maxsize=100)

    # # 自测本地vggt图片文件夹模拟队列输入
    # video_thread = threading.Thread(
    #     target=load_images_to_queue,
    #     args=(f"{base_url}/rgb", frame_queue),
    #     daemon=True
    # )
    # video_thread.start()

    # 自测本地视频/图片模拟队列输入
    video_thread = threading.Thread(
        target=load_media_to_queue,
        # args=("/home/ubuntu/Downloads/normal_video15/rgb", frame_queue),
        args=("/home/ubuntu/datasets/real_data/video_2.avi", frame_queue),
        # args=("/home/ubuntu/datasets/coco/images/test", frame_queue),
        # args=("/home/ubuntu/datasets/video/realsense_output_6.avi", frame_queue),
        # args=("results/test.avi", frame_queue),
        # args=("results/realsense_output.avi", frame_queue),
        daemon=True
    )
    video_thread.start()

    # 创建客户端线程
    client = KonkaGrpcClient(frame_queue=frame_queue, result_queue=result_queue)
    client_thread = threading.Thread(
        target=client.run,
        daemon=True
    )
    client_thread.start()

    # 获取结果样例
    try:
        while True:
            try:
                tracks, timestamp = client.result_queue.get(timeout=1)
                # append_tracking_result_scannet(tracks, timestamp, base_url)
                # append_tracking_result_vggt(tracks, timestamp, base_url, f"{base_url}/scene_info_meta.json")
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        # # 最后保存所有结果
        # with open(f"{base_url}/scene_info_meta.json", 'w') as f:
        #     json.dump(tracking_results, f, indent=2)
        #     logger.info(f"已保存结果到 {base_url}/scene_info_meta.json")
