import os
import time
import threading
import queue
import yaml
import grpc
import numpy as np
import cv2

import imgs_pb2
import imgs_pb2_grpc
from utils.logger import SingletonLogger

# 配置日志系统
logger = SingletonLogger(
    name='client',
    log_file='logs/client.log',
    level='INFO',
    console=True
)


class KonkaGrpcClient:
    def __init__(self, result_queue=None):
        default_config_path = "client_config.yaml"
        config = {}

        # 尝试加载 YAML 配置文件
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {default_config_path}")
        else:
            logger.warning(f"Config file not found, using defaults.")

        self.cfg = config
        self.server_ip = config.get('server_ip', '127.0.0.1')
        self.server_port = config.get('server_port', '50051')
        self.data_size = config.get('data_size', 500)

        # 可视化配置 (在黑底上画框，用于测试)
        self.visualize = True
        self.width = 640  # 默认画布宽
        self.height = 480  # 默认画布高

        self.result_queue = result_queue

    def request_iterator(self):
        """生成器：仅用于建立连接和维持会话"""
        try:
            logger.info("Sending initial handshake...")
            # 发送一个空的包，带上当前时间戳
            yield imgs_pb2.Img(
                img=b'',
                timeStamp=str(time.time()).encode("utf-8"),
                requiresMasks=True
            )

            while True:
                time.sleep(10)
        except Exception as e:
            logger.error(f"Request iterator stop: {e}")

    def draw_on_black_canvas(self, tracks, timestamp_str):
        # 创建黑底画布
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        cv2.putText(canvas, f"TS: {timestamp_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for track in tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            track_id = track[4]
            conf = track[5]
            cls_name = track[6]
            mask_coords = track[7]

            # 生成随机颜色
            np.random.seed(int(track_id) if isinstance(track_id, int) else hash(track_id) % 255)
            color = tuple(np.random.randint(50, 255, 3).tolist())

            # 1. 画框
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # 2. 画标签
            label = f"{cls_name} {track_id} {conf:.2f}"
            cv2.putText(canvas, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # 3. 画掩码 (Mask)
            if mask_coords and len(mask_coords) > 0:
                pts = np.array(mask_coords, dtype=np.int32).reshape((-1, 1, 2))
                overlay = canvas.copy()
                cv2.drawContours(overlay, [pts], -1, color, -1)  # 填充
                cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
                cv2.drawContours(canvas, [pts], -1, (255, 255, 255), 1)

        cv2.imshow("Client Test View (No Source Image)", canvas)
        cv2.waitKey(1)

    def run(self):
        options = [
            ('grpc.max_send_message_length', self.data_size * 1024 * 1024),
            ('grpc.max_receive_message_length', self.data_size * 1024 * 1024)
        ]

        with grpc.insecure_channel(f"{self.server_ip}:{self.server_port}", options=options) as channel:
            logger.info(f"Connected to {self.server_ip}:{self.server_port}")

            stub = imgs_pb2_grpc.ImageProcessingStub(channel)

            try:
                # 调用服务端流式接口
                responses = stub.continuousInfo(self.request_iterator())

                logger.info("Waiting for stream data from server...")

                for response in responses:
                    server_ts_str = response.timeStamp.decode("utf-8")

                    try:
                        server_ts = float(server_ts_str)
                        latency = time.time() - server_ts
                    except:
                        latency = 0.0

                    tracks_decoded = []
                    if response.tracks:
                        for t in response.tracks:
                            mask_points = []
                            for m in t.mask:
                                mask_points.append([m.x, m.y])

                            tracks_decoded.append([
                                t.x1, t.y1, t.x2, t.y2,  # 0-3
                                t.trackId,  # 4
                                t.conf,  # 5
                                t.cls,  # 6
                                mask_points,  # 7
                                t.isMove,  # 8
                                list(t.feats)  # 9
                            ])

                    logger.info(
                        f"Frame TS: {server_ts_str} | Latency: {latency * 1000:.1f}ms | Objects: {len(tracks_decoded)}")

                    if self.result_queue:
                        self.result_queue.put((tracks_decoded, server_ts_str))

                    if self.visualize:
                        self.draw_on_black_canvas(tracks_decoded, server_ts_str)

            except grpc.RpcError as e:
                logger.error(f"gRPC disconnected: {e.code()} - {e.details()}")
            except KeyboardInterrupt:
                logger.info("Stopping client...")
            finally:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    client = KonkaGrpcClient(result_queue=None)
    client.run()
