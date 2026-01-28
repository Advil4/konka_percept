import os.path
import time
from concurrent import futures

import cv2
import grpc
import numpy as np
import yaml

import imgs_pb2
import imgs_pb2_grpc
import pipeline
from utils.helpers import get_color, load_vocab
from utils.logger import SingletonLogger

# 配置日志系统
logger = SingletonLogger(
    name='pipeline',
    log_file='logs/pipeline.log',
    level='INFO',
    console=True
)


class ImageProcessingServicer(imgs_pb2_grpc.ImageProcessingServicer):
    def __init__(self, pipe_line, cfg=None):
        self.pipe = pipe_line
        self.pipe.start_threads()
        self.pipeline_initialized = False
        self.current_task = None

        # 从配置文件读取可视化参数
        if cfg and 'visual_cfg' in cfg:
            visual_cfg = cfg['visual_cfg']
            self.visual = visual_cfg.get('visual')
            self.video_save = visual_cfg.get('video_save')
            self.img_show = visual_cfg.get('img_show')
            self.img_save = visual_cfg.get('img_save')
            self.output_path = visual_cfg.get('output_path')
            self.images_path = visual_cfg.get('images_path')

        self.vocab = load_vocab()
        # 原始图像默认尺寸
        self.original_width, self.original_height = 480, 640
        # self.original_width, self.original_height = 378, 672

        # 计算 padding（取图像宽高中较大值的 20%）
        self.padding = int(max(self.original_width, self.original_height) * 0.4)

        # 扩展后的尺寸
        self.extended_width = self.original_width + 2 * self.padding
        self.extended_height = self.original_height + 2 * self.padding
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (self.extended_width, self.extended_height))

        logger.info(f"image processing servicer initialized")

    def continuousInfo(self, request_iterator, context):
        """
        处理连续信息流，提供基于图像的跟踪信息。

        参数:
            request_iterator: 请求迭代器，包含一系列图像请求。
            context: 上下文信息，未在本函数中使用。

        返回:
            生成器， yield 包含跟踪信息的响应。
        """
        self.tracking_history = {}

        for request in request_iterator:

            # 解码图像和时间戳
            color_img = request.img
            time_stamp = float(request.timeStamp.decode("utf-8"))
            img = np.frombuffer(color_img, dtype=np.uint8).reshape(self.original_height, self.original_width, 3)

            # 判断任务类型
            task_type = 'segment' if request.requiresMasks else 'detect'

            # 初始化模型类（首次或切换任务时）
            if not self.pipeline_initialized or self.current_task != task_type:
                self.pipe.init_predictor(task_type)
                self.pipeline_initialized = True
                self.current_task = task_type

            # 设置输入帧
            self.pipe.set_source((img, time_stamp))

            # 执行推理流水线
            start_time = time.time()
            data = self.pipe.run_pipeline()
            all_time = (time.time() - start_time) * 1e3
            if data is None:
                logger.warning(f"No data returned for time_stamp: {time_stamp}, skipping...")
                continue
            logger.info(
                f"pre_time: {data.pre_time:.2f}ms infer_time: {data.infer_time:.2f}ms post_time: {data.post_time:.2f}ms "
                f"track_time: {data.track_time:.2f} result_time: {data.result_time:.2f}ms all_time: {all_time:.2f}ms")

            # 提取结果
            tracked_objects = data.tracked_objects
            if len(tracked_objects) == 0:
                logger.warning("No_tracked_objects")
            if data.masks_coords is not None:
                masks_coords = np.array(data.masks_coords, dtype=np.int32)
            else:
                masks_coords = []
            if data.mask_sizes is not None:
                mask_sizes = np.array(data.mask_sizes, dtype=np.uint16)
            else:
                mask_sizes = []

            # 服务端可视化保存
            if self.visual:
                if not os.path.exists(self.output_path):
                    self.writer = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'MJPG'), 10,
                                                  (self.extended_width, self.extended_height))
                vis_data = {
                    "img": img.copy(),
                    "tracked_objects": tracked_objects.copy(),
                    "masks_coords": masks_coords.copy(),
                    "mask_sizes": mask_sizes.copy()
                }

                img = vis_data.get("img")
                tracked_objects_visual = vis_data.get("tracked_objects")
                masks_coords_visual = vis_data.get("masks_coords")
                mask_sizes_visual = vis_data.get("mask_sizes")

                need_mask = False
                if masks_coords_visual is not None:
                    need_mask = True

                # 计算带扩展区域的画布大小
                padding = self.padding
                new_width = img.shape[1] + 2 * padding
                new_height = img.shape[0] + 2 * padding

                # 创建新画布并把原图放在中间
                extended_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
                extended_img[padding:padding + img.shape[0], padding:padding + img.shape[1]] = img

                if need_mask:
                    for track, mask_size in zip(tracked_objects_visual, mask_sizes_visual):
                        x1, y1, x2, y2 = map(round, track[:4])
                        track_id = track[4]
                        conf = track[5]
                        cls_id = int(track[6])
                        color = get_color(cls_id)
                        is_move = False

                        # 调整坐标以适配新画布
                        x1 += padding
                        y1 += padding
                        x2 += padding
                        y2 += padding

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = max(0, x2), max(0, y2)

                        if is_move:
                            label = f"* {self.vocab[cls_id]}_{conf:.2f}_{track_id}" if cls_id < len(
                                self.vocab) else f"ID:{track_id}"
                        else:
                            label = f"{self.vocab[cls_id]}_{conf:.2f}_{track_id}" if cls_id < len(
                                self.vocab) else f"ID:{track_id}"

                        # 绘制矩形框
                        cv2.rectangle(extended_img, (x1, y1), (x2, y2), color, 2)

                        # 绘制标签文字
                        cv2.putText(extended_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 2)

                        # 绘制mask部分
                        if masks_coords_visual is not None and mask_size > 0:
                            contour = masks_coords_visual[:mask_size]
                            masks_coords_visual = masks_coords_visual[mask_size:]
                            contour = np.array(contour).reshape(-1, 1, 2)

                            # 调整mask坐标
                            contour[:, :, 0] += padding  # x 坐标偏移
                            contour[:, :, 1] += padding  # y 坐标偏移

                            mask = np.zeros(extended_img.shape[:2], dtype=np.uint8)
                            cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)
                            mask_ = (mask > 0.5).astype(np.uint8)
                            extended_img[mask_ == 1] = extended_img[mask_ == 1] * 0.5 + np.array(
                                color) * 0.5
                else:
                    for track in tracked_objects_visual:
                        x1, y1, x2, y2 = map(round, track[:4])
                        track_id = track[4]
                        conf = track[5]
                        cls_id = int(track[6])
                        color = get_color(cls_id)

                        # 调整坐标以适配新画布
                        x1 += padding
                        y1 += padding
                        x2 += padding
                        y2 += padding

                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = max(0, x2), max(0, y2)

                        label = f"{self.vocab[cls_id]}_{conf:.2f}_{track_id}" if cls_id < len(
                            self.vocab) else f"ID:{track_id}"
                        cv2.rectangle(extended_img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(extended_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    color, 2)

                if self.img_show:
                    cv2.imshow("Result", extended_img)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break

                if self.video_save:
                    self.writer.write(extended_img)
                if self.img_save:
                    timestamp_str = request.timeStamp.decode("utf-8")
                    timestamp = "{0:.6f}".format(float(timestamp_str))
                    if not os.path.exists(self.images_path):
                        os.makedirs(self.images_path)
                    cv2.imwrite(f"{self.images_path}/{timestamp}.png", extended_img)
                logger.info(f"frame written to {self.output_path}")

            # grpc传输数据结构
            tracks = []
            if len(masks_coords) > 0 and masks_coords is not None and tracked_objects is not None:
                for obj, mask_size in zip(tracked_objects, mask_sizes):
                    track_row = imgs_pb2.TrackRow()
                    track_row.x1 = obj[0]
                    track_row.y1 = obj[1]
                    track_row.x2 = obj[2]
                    track_row.y2 = obj[3]
                    track_row.trackId = obj[4]
                    track_row.conf = obj[5]
                    track_row.cls = self.vocab[int(obj[6])]
                    track_row.isMove = False
                    track_row.feats.extend(obj[9])

                    mask = []
                    if masks_coords is not None and mask_size > 0:
                        contour = masks_coords[:mask_size]
                        masks_coords = masks_coords[mask_size:]
                        for contour_coordinate in contour:
                            x = int(contour_coordinate[0]) if contour_coordinate[0] is not None else 0
                            y = int(contour_coordinate[1]) if contour_coordinate[1] is not None else 0
                            mask_row = imgs_pb2.Masks()
                            mask_row.x = x
                            mask_row.y = y
                            mask.append(mask_row)
                    for m in mask:
                        track_row.mask.add().CopyFrom(m)
                    tracks.append(track_row)
            # 发送响应
            yield imgs_pb2.MasksTracks(
                timeStamp=request.timeStamp,
                tracks=tracks,
            )


def serve(cfg, pipe_line):
    """
    启动基于 gRPC 的图像处理服务。

    参数:
        cfg: 服务器配置字典，包含最大线程数、消息长度限制、IP地址和端口号等信息。
        pipe_line: 图像处理流水线对象，定义了图像推理和处理流程。

    异常:
        如果 [cfg](file:///home/ubuntu/Project/konkaV2.0.0/pipeline.py#L0-L0) 或 `pipe_line` 为 None，
        则抛出 ValueError 表示缺少必要参数。

    返回值:
        无返回值。该函数启动一个持续运行的 gRPC 服务器。
    """
    # 检查配置是否为空，若为空则抛出异常
    if cfg is None:
        raise ValueError("cfg is None")
    # 检查流水线是否为空，若为空则抛出异常
    if pipe_line is None:
        raise ValueError("a pipe_line is needed")

    # 创建 gRPC 服务器，使用线程池执行器和自定义选项
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=cfg["server_cfg"]["max_workers"]),
                         options=[
                             ('grpc.max_send_message_length', cfg["server_cfg"]["max_send_message_length"]),
                             ('grpc.max_receive_message_length', cfg["server_cfg"]["max_receive_message_length"])
                         ])

    # 注册图像处理服务到 gRPC 服务器
    imgs_pb2_grpc.add_ImageProcessingServicer_to_server(
        ImageProcessingServicer(pipe_line, cfg), server
    )

    # 绑定监听地址和端口
    server.add_insecure_port(cfg["server_cfg"]["server_ip"] + ":" + cfg["server_cfg"]["server_port"])

    # 启动服务器
    server.start()
    logger.info(f"server initiated")

    # 等待客户端连接与请求，阻塞直到服务终止
    server.wait_for_termination()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)  # suppress参数用于禁用科学计数法
    with open("server_config.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    # 流水线类实例化
    s_time = time.time()
    pipe = pipeline.PipelinePredictor(cfg=cfg)
    e_time = time.time()
    logger.info(f"Pipeline initialized in {(e_time - s_time):.2f}ms")

    serve(cfg, pipe)
