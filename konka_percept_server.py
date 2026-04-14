import json
import queue
import signal
import sys
import threading
import time
import os
from concurrent import futures

import cv2
import grpc
import numpy as np
import yaml

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("Warning: ROS2 not found. Server will fail to subscribe.")

import imgs_pb2
import imgs_pb2_grpc
import pipeline
from utils.helpers import get_color, load_vocab
from utils.logger import SingletonLogger

logger = SingletonLogger(
    name='pipeline',
    log_file='logs/pipeline.log',
    level='INFO',
    console=True
)

STOP_EVENT = threading.Event()


class RosImageSource(threading.Thread):
    def __init__(self, frame_queue, cfg=None):
        super().__init__(name="RosThread")
        self.frame_queue = frame_queue
        self.topic = cfg["ros_cfg"]["topic"]
        self.target_fps = cfg["ros_cfg"]["target_fps"]
        self.interval = 1.0 / self.target_fps if self.target_fps > 0 else 0
        self.last_accept_time = 0
        self.node = None
        self.daemon = True

    def run(self):
        if not ROS_AVAILABLE:
            return
        try:
            self.node = Node("perception_grpc_bridge")
            bridge = CvBridge()

            def callback(msg):
                if STOP_EVENT.is_set():
                    return

                current_time = time.time()
                if (current_time - self.last_accept_time) < self.interval:
                    return
                self.last_accept_time = current_time

                try:
                    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                    # 逆时针旋转90度
                    cv_img = cv2.rotate(cv_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass

                    self.frame_queue.put((cv_img, ts))

                except Exception as e:
                    logger.error(f"ROS Callback Error: {e}")

            self.node.create_subscription(Image, self.topic, callback, 1)
            logger.info(
                f"ROS Subscribed to {self.topic} (Rate Limit: {self.target_fps} FPS)"
            )

            while not STOP_EVENT.is_set() and rclpy.ok():
                rclpy.spin_once(self.node, timeout_sec=0.1)

        except Exception as e:
            logger.error(f"ROS Thread Error: {e}")
        finally:
            if self.node:
                self.node.destroy_node()
            logger.info("ROS Thread stopped.")


class ImageProcessingServicer(imgs_pb2_grpc.ImageProcessingServicer):
    def __init__(self, pipe_line, input_queue, cfg):
        self.pipe = pipe_line
        self.input_queue = input_queue
        self.pipeline_initialized = False

        self.client_queues = []
        self.client_lock = threading.Lock()

        # 加载词汇表和配置
        self.vocab = load_vocab()
        visual_cfg = cfg["visual_cfg"]
        self.video_save = visual_cfg.get("video_save")
        img_show = visual_cfg.get("img_show")
        img_save = visual_cfg.get("img_save")
        img_result_save = visual_cfg.get("img_result_save")
        json_result_save = visual_cfg.get("json_result_save")
        self.video_path = visual_cfg.get("video_path")
        images_path = visual_cfg.get("images_path")
        images_result_path = visual_cfg.get("images_result_path")
        json_result_path = visual_cfg.get("json_result_path")

        # 全局控制开关
        self.percept_enable = True
        self.percept_enable_visual = img_show
        self.percept_enable_img_save = img_save
        self.percept_enable_visual_save = img_result_save
        self.percept_enable_result_save = json_result_save

        # 路径配置
        self.percept_enable_img_save_path = images_path
        self.percept_enable_visual_save_path = images_result_path
        self.percept_enable_result_save_path = json_result_path

        self.original_width, self.original_height = 640, 480
        self.padding = 0
        self.writer = None
        if self.video_save:
            self.fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        self.all_results = {}

        # 启动唯一的后台处理线程
        self.worker_thread = threading.Thread(
            target=self._process_pipeline_loop, name="PipelineWorker"
        )
        self.worker_thread.daemon = True
        self.worker_thread.start()

    def _init_writer_if_needed(self, width, height):
        if self.video_save and self.writer is None:
            self.padding = int(max(width, height) * 0)
            self.extended_width = width + 2 * self.padding
            self.extended_height = height + 2 * self.padding
            self.writer = cv2.VideoWriter(
                self.video_path,
                self.fourcc,
                10,
                (self.extended_width, self.extended_height),
            )

    def _broadcast(self, response_msg):
        """将结果广播给所有连接的客户端"""
        with self.client_lock:
            for client_q in self.client_queues:
                if client_q.full():
                    try:
                        client_q.get_nowait()
                    except queue.Empty:
                        pass
                client_q.put(response_msg)

    def _process_pipeline_loop(self):
        """后台推理与分发线程"""
        task_type = "segment"
        if not self.pipeline_initialized:
            self.pipe.init_predictor(task_type)
            self.pipeline_initialized = True

        logger.info("Pipeline worker thread started.")

        try:
            while not STOP_EVENT.is_set():
                try:
                    img, timestamp = self.input_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                ts_str = "{:.6f}".format(timestamp)

                # 1. 如果感知被关闭，清空图像并广播空数据以维持心跳
                if not self.percept_enable:
                    response_msg = imgs_pb2.MasksTracks(
                        timeStamp=ts_str.encode("utf-8"), tracks=[]
                    )
                    self._broadcast(response_msg)
                    continue

                # 更新尺寸并执行推理
                h, w = img.shape[:2]
                self._init_writer_if_needed(w, h)
                self.pipe.set_source((img, timestamp))
                start_time = time.time()
                data = self.pipe.run_pipeline()
                all_time = (time.time() - start_time) * 1e3

                if data is None:
                    logger.warning(f"No data returned for time_stamp: {timestamp}, skipping...")
                    continue
                logger.info(
                    f"pre_time: {data.pre_time:.2f}ms infer_time: {data.infer_time:.2f}ms post_time: {data.post_time:.2f}ms "
                    f"track_time: {data.track_time:.2f}ms result_time: {data.result_time:.2f}ms all_time: {all_time:.2f}ms")

                tracked_objects = data.tracked_objects
                masks_coords = (
                    np.array(data.masks_coords, dtype=np.int32)
                    if data.masks_coords
                    else []
                )
                mask_sizes = (
                    np.array(data.mask_sizes, dtype=np.uint16)
                    if data.mask_sizes
                    else []
                )

                # 2. 可视化模块
                if self.percept_enable_visual:
                    vis_img = img.copy()
                    padding = self.padding
                    new_w, new_h = w + 2 * padding, h + 2 * padding
                    extended_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    extended_img[padding: padding + h, padding: padding + w] = vis_img

                    if len(tracked_objects) > 0:
                        for track, mask_size in zip(tracked_objects, mask_sizes):
                            x1, y1, x2, y2 = map(round, track[:4])
                            cls_id = int(track[6])
                            color = get_color(cls_id)
                            x1, y1 = max(0, x1 + padding), max(0, y1 + padding)
                            x2, y2 = max(0, x2 + padding), max(0, y2 + padding)

                            cv2.rectangle(extended_img, (x1, y1), (x2, y2), color, 2)

                            if mask_size > 0:
                                contour = masks_coords[:mask_size]
                                masks_coords = masks_coords[mask_size:]
                                contour = np.array(contour).reshape(-1, 1, 2)
                                contour[:, :, 0] += padding
                                contour[:, :, 1] += padding
                                mask_canvas = np.zeros(
                                    extended_img.shape[:2], dtype=np.uint8
                                )
                                cv2.drawContours(
                                    mask_canvas, [contour], -1, 1, thickness=cv2.FILLED
                                )
                                mask_bool = (mask_canvas > 0.5).astype(np.uint8)
                                extended_img[mask_bool == 1] = (
                                        extended_img[mask_bool == 1] * 0.5
                                        + np.array(color) * 0.5
                                )

                    # 显示弹窗
                    if self.percept_enable_visual:
                        cv2.imshow("Server Visual", extended_img)
                        if cv2.waitKey(1) == ord("q"):
                            STOP_EVENT.set()

                    # 3. 图像与可视化结果本地保存
                    if self.video_save and self.writer:
                        self.writer.write(extended_img)

                    # 支持动态模型保存
                    if self.percept_enable_img_save:
                        os.makedirs(self.percept_enable_img_save_path, exist_ok=True)
                        cv2.imwrite(
                            f"{self.percept_enable_img_save_path}/{ts_str}.png", vis_img
                        )

                    if self.percept_enable_visual_save:
                        os.makedirs(self.percept_enable_visual_save_path, exist_ok=True)
                        cv2.imwrite(
                            f"{self.percept_enable_visual_save_path}/{ts_str}.png",
                            extended_img,
                        )
                else:
                    cv2.destroyAllWindows()

                masks_coords = (
                    np.array(data.masks_coords, dtype=np.int32)
                    if data.masks_coords
                    else []
                )

                # 4. grpc传输数据结构封装
                tracks = []
                if len(masks_coords) > 0 and tracked_objects is not None:
                    for obj, mask_size in zip(tracked_objects, mask_sizes):
                        track_row = imgs_pb2.TrackRow(
                            x1=obj[0],
                            y1=obj[1],
                            x2=obj[2],
                            y2=obj[3],
                            trackId=str(obj[4]),
                            conf=obj[5],
                            cls=self.vocab[int(obj[6])],
                            isMove=False,
                        )
                        track_row.feats.extend(obj[9])

                        if mask_size > 0:
                            contour = masks_coords[:mask_size]
                            masks_coords = masks_coords[mask_size:]
                            for pt in contour:
                                track_row.mask.add(
                                    x=int(pt[0]),
                                    y=int(pt[1]),
                                )
                        tracks.append(track_row)

                    # 本地保存感知结果数据
                    if self.percept_enable_result_save:
                        self.all_results[ts_str] = []
                        for track in tracks:
                            self.all_results[ts_str].append(
                                {
                                    "x1": track.x1,
                                    "y1": track.y1,
                                    "x2": track.x2,
                                    "y2": track.y2,
                                    "trackId": track.trackId,
                                    "conf": track.conf,
                                    "cls": track.cls,
                                    "feats": list(track.feats),
                                    "mask": [[m.x, m.y] for m in track.mask],
                                }
                            )
                        try:
                            os.makedirs(
                                self.percept_enable_result_save_path, exist_ok=True
                            )
                            with open(
                                    f"{self.percept_enable_result_save_path}/all_results.json",
                                    "w",
                                    encoding="utf-8",
                            ) as f:
                                json.dump(
                                    self.all_results, f, indent=2, ensure_ascii=False
                                )
                        except Exception as e:
                            logger.error(f"Save JSON Error：{e}")

                # 5. 结果广播
                response_msg = imgs_pb2.MasksTracks(
                    timeStamp=ts_str.encode("utf-8"), tracks=tracks
                )
                self._broadcast(response_msg)

        except Exception as e:
            logger.error(f"Worker Error: {e}")
        finally:
            if self.writer:
                self.writer.release()
                self.writer = None

    def continuousInfo(self, request_iterator, context):
        """
        处理客户端连接：
        - 开启子线程监听指令 (request_iterator)
        - 主线程负责把结果通过 yield 发送给客户端
        """
        logger.info(f"Client connected. Peer: {context.peer()}")

        # 为当前客户端创建一个最大容量为10的专属缓冲队列
        my_queue = queue.Queue(maxsize=10)

        with self.client_lock:
            self.client_queues.append(my_queue)

        # 异步读取客户端指令并给予状态应答
        def read_commands():
            try:
                for req in request_iterator:
                    status_list = []

                    # 1. 优先处理感知开关指令 (优先保证状态同步)
                    if req.HasField("percept_enable"):
                        try:
                            self.percept_enable = req.percept_enable
                            logger.info(
                                f"Command: percept_enable -> {self.percept_enable}"
                            )
                            status_list.append(
                                imgs_pb2.CommandStatus(
                                    command="percept_enable", message="success"
                                )
                            )
                        except Exception:
                            status_list.append(
                                imgs_pb2.CommandStatus(
                                    command="percept_enable", message="failure"
                                )
                            )

                    # 处理布尔类型指令
                    def process_bool_cmd(field_name):
                        if req.HasField(field_name):
                            val = getattr(req, field_name)
                            # 在感知被关闭的情况下，其他功能直接返回 failure
                            if not self.percept_enable:
                                logger.warning(
                                    f"Rejected: Cannot enable {field_name} because percept_enable is False."
                                )
                                status_list.append(
                                    imgs_pb2.CommandStatus(
                                        command=field_name, message="failure"
                                    )
                                )
                            else:
                                setattr(self, field_name, val)
                                logger.info(f"Command: {field_name} -> {val}")
                                status_list.append(
                                    imgs_pb2.CommandStatus(
                                        command=field_name, message="success"
                                    )
                                )

                    # 处理字符串类型(路径配置)指令
                    def process_str_cmd(field_name):
                        if req.HasField(field_name):
                            # 感知关闭时，不允许修改任何保存路径配置
                            if not self.percept_enable:
                                logger.warning(
                                    f"Rejected: Cannot set {field_name} because percept_enable is False."
                                )
                                status_list.append(
                                    imgs_pb2.CommandStatus(
                                        command=field_name, message="failure"
                                    )
                                )
                            else:
                                val = getattr(req, field_name)
                                setattr(self, field_name, val)
                                logger.info(f"Command: {field_name} -> {val}")
                                status_list.append(
                                    imgs_pb2.CommandStatus(
                                        command=field_name, message="success"
                                    )
                                )

                    # 2. 处理各项功能开关
                    process_bool_cmd("percept_enable_visual")
                    process_bool_cmd("percept_enable_img_save")
                    process_bool_cmd("percept_enable_visual_save")
                    process_bool_cmd("percept_enable_result_save")

                    # 3. 处理各项路径配置 (需配合 string 修改)
                    process_str_cmd("percept_enable_img_save_path")
                    process_str_cmd("percept_enable_visual_save_path")
                    process_str_cmd("percept_enable_result_save_path")

                    # 4. 如果本次请求包含任何指令，则生成一条 ACK 回执发给该客户端
                    if status_list:
                        ts_str = "{:.6f}".format(time.time())
                        ack_msg = imgs_pb2.MasksTracks(
                            timeStamp=ts_str.encode("utf-8"),
                            tracks=[],  # 纯状态应答，不带图像数据
                            command_statuses=status_list,
                        )

                        # 塞入当前客户端自己的专属队列
                        if my_queue.full():
                            try:
                                my_queue.get_nowait()
                            except queue.Empty:
                                pass
                        my_queue.put(ack_msg)

            except Exception as e:
                logger.debug(f"Command listener stopped. Reason: {e}")

        # 启动监听指令的后台线程
        cmd_thread = threading.Thread(target=read_commands, daemon=True)
        cmd_thread.start()

        # 主循环：持续向客户端发送结果流
        try:
            while context.is_active() and not STOP_EVENT.is_set():
                try:
                    response = my_queue.get(timeout=0.5)
                    yield response
                except queue.Empty:
                    continue
        except Exception as e:
            logger.error(f"gRPC Stream Error: {e}")
        finally:
            logger.info(f"Client disconnected. Peer: {context.peer()}")
            with self.client_lock:
                if my_queue in self.client_queues:
                    self.client_queues.remove(my_queue)


def main():
    with open("server_config.yaml", "r") as file:
        cfg = yaml.safe_load(file)
    frame_queue = queue.Queue(maxsize=5)

    # 初始化 Pipeline
    logger.info("Initializing Pipeline...")
    pipe = pipeline.PipelinePredictor(cfg=cfg)
    pipe.start_threads()

    # 启动 ROS 线程
    if ROS_AVAILABLE:
        rclpy.init(args=None)
        ros_thread = RosImageSource(frame_queue, cfg=cfg)
        ros_thread.start()
    else:
        logger.warning("Starting without ROS libs.")

    # 启动 gRPC Server
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=cfg["server_cfg"]["max_workers"]),
        options=[
            (
                "grpc.max_send_message_length",
                cfg["server_cfg"]["max_send_message_length"],
            ),
            (
                "grpc.max_receive_message_length",
                cfg["server_cfg"]["max_receive_message_length"],
            ),
        ],
    )

    servicer = ImageProcessingServicer(pipe, frame_queue, cfg)
    imgs_pb2_grpc.add_ImageProcessingServicer_to_server(servicer, server)

    port = cfg["server_cfg"]["server_port"]
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"gRPC Server listening on {port}")

    def signal_handler(sig, frame):
        logger.warning("🛑 Shutting down...")
        STOP_EVENT.set()

        server.stop(grace=1).wait(1)

        if ROS_AVAILABLE and rclpy.ok():
            rclpy.shutdown()

        pipe.stop_threads()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while not STOP_EVENT.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    main()
