import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraConfig:
    """摄像头配置类"""

    CAMERA_TYPE_ORBBEC = "orbbec"
    CAMERA_TYPE_REALSENSE = "realsense"

    def __init__(self):
        self.camera_type = self.CAMERA_TYPE_REALSENSE  # 选择摄像头类型: "orbbec" 或 "realsense"

        # ROS2 发布配置
        self.topic_name = '/head_camera/color/image_raw'
        self.frame_id = 'camera_color_frame'
        self.publish_fps = 30.0
        self.queue_size = 10

        self.orbbec_width = 640
        self.orbbec_height = 480
        self.orbbec_format = 'RGB888'  # 可选: RGB888, MJPG, YUYV
        self.orbbec_fps = 30

        self.realsense_width = 640
        self.realsense_height = 480
        self.realsense_format = 'RGB8'  # 可选: RGB8, BGR8
        self.realsense_fps = 30
        self.realsense_enable_auto_exposure = True
        self.realsense_enable_auto_white_balance = True


class OrbbecCameraBackend:
    """Orbbec 摄像头后端"""

    def __init__(self, config, logger):
        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

        self.logger = logger
        self.pipeline = Pipeline()
        self.sdk_config = Config()
        self.OBSensorType = OBSensorType
        self.OBFormat = OBFormat

        try:
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is None:
                raise Exception("未找到彩色传感器，请检查连接！")

            try:
                format_map = {
                    'RGB888': OBFormat.RGB888,
                    'MJPG': OBFormat.MJPG,
                    'YUYV': OBFormat.YUYV
                }
                ob_format = format_map.get(config.orbbec_format, OBFormat.RGB888)
                profile = profile_list.get_video_stream_profile(
                    config.orbbec_width,
                    config.orbbec_height,
                    ob_format,
                    config.orbbec_fps
                )
            except:
                profile = profile_list.get_default_video_stream_profile()
                self.logger.warn(f"使用默认配置: {profile}")

            self.sdk_config.enable_stream(profile)
            self.pipeline.start(self.sdk_config)
            self.logger.info(
                f"✅ Orbbec 摄像头已启动 [{config.orbbec_width}x{config.orbbec_height}@{config.orbbec_fps}fps]")

        except Exception as e:
            self.logger.error(f"❌ Orbbec 摄像头启动失败: {e}")
            raise

    def get_frame(self):
        """获取一帧图像，返回 BGR 格式的 numpy 数组"""
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        data = np.asanyarray(color_frame.get_data())
        width = color_frame.get_width()
        height = color_frame.get_height()

        if data.size != width * height * 3:
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img_bgr is None:
                self.logger.warn("解码图像失败")
                return None
        else:
            img_rgb = data.reshape((height, width, 3))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return img_bgr

    def stop(self):
        """停止摄像头"""
        self.pipeline.stop()


class RealSenseCameraBackend:
    """RealSense 摄像头后端"""

    def __init__(self, config, logger):
        import pyrealsense2 as rs

        self.logger = logger
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs = rs

        try:
            format_map = {
                'RGB8': rs.format.rgb8,
                'BGR8': rs.format.bgr8
            }
            rs_format = format_map.get(config.realsense_format, rs.format.rgb8)

            self.rs_config.enable_stream(
                rs.stream.color,
                config.realsense_width,
                config.realsense_height,
                rs_format,
                config.realsense_fps
            )

            profile = self.pipeline.start(self.rs_config)

            if config.realsense_enable_auto_exposure:
                color_sensor = profile.get_device().first_color_sensor()
                color_sensor.set_option(rs.option.enable_auto_exposure, True)
                color_sensor.set_option(rs.option.enable_auto_white_balance, True)

            self.logger.info(
                f"✅ RealSense 摄像头已启动 [{config.realsense_width}x{config.realsense_height}@{config.realsense_fps}fps]")

        except Exception as e:
            self.logger.error(f"❌ RealSense 摄像头启动失败: {e}")
            raise

    def get_frame(self):
        """获取一帧图像，返回 BGR 格式的 numpy 数组"""
        frames = self.pipeline.wait_for_frames(1000)
        if frames is None:
            return None

        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None

        color_image = np.asanyarray(color_frame.get_data())

        if color_frame.get_profile().format() == self.rs.format.rgb8:
            img_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = color_image

        return img_bgr

    def stop(self):
        """停止摄像头"""
        self.pipeline.stop()


class CameraImagePublisher(Node):
    """通用摄像头图像发布器（支持 Orbbec 和 RealSense）"""

    def __init__(self):
        super().__init__('camera_rgb_publisher')

        self.config = CameraConfig()
        self.bridge = CvBridge()

        self.publisher_ = self.create_publisher(
            Image,
            self.config.topic_name,
            self.config.queue_size
        )

        self.total_published_frames = 0  # 总发布帧数
        self.interval_frames = 0  # 间隔内发布的帧数 (用于计算实时FPS)
        self.last_print_time = time.time()

        self.camera_backend = self._init_camera()

        if self.camera_backend is not None:
            self.timer = self.create_timer(
                1.0 / self.config.publish_fps,
                self.timer_callback
            )
            self.get_logger().info(f"📷 摄像头发布器已启动，发布到: {self.config.topic_name}")
        else:
            self.get_logger().error("❌ 摄像头初始化失败，节点将退出")

    def _init_camera(self):
        """根据配置初始化对应的摄像头后端"""
        try:
            if self.config.camera_type == CameraConfig.CAMERA_TYPE_ORBBEC:
                self.get_logger().info("🔄 初始化 Orbbec 摄像头...")
                return OrbbecCameraBackend(self.config, self.get_logger())

            elif self.config.camera_type == CameraConfig.CAMERA_TYPE_REALSENSE:
                self.get_logger().info("🔄 初始化 RealSense 摄像头...")
                return RealSenseCameraBackend(self.config, self.get_logger())

            else:
                self.get_logger().error(f"❌ 不支持的摄像头类型: {self.config.camera_type}")
                return None

        except Exception as e:
            self.get_logger().error(f"❌ 摄像头初始化异常: {e}")
            return None

    def timer_callback(self):
        """定时回调：获取帧并发布"""
        try:
            img_bgr = self.camera_backend.get_frame()

            if img_bgr is None:
                return

            # 转成 ROS 消息并发布
            msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.config.frame_id
            self.publisher_.publish(msg)

            self.total_published_frames += 1
            self.interval_frames += 1

            current_time = time.time()
            elapsed_time = current_time - self.last_print_time

            # 每隔 2 秒打印一次状态流水
            if elapsed_time >= 2.0:
                actual_fps = self.interval_frames / elapsed_time
                self.get_logger().info(
                    f"🌊 [流水线正常] 总计发布: {self.total_published_frames:05d} 帧 | "
                    f"实时输出帧率: {actual_fps:.1f} FPS"
                )

                # 重置间隔计数器
                self.interval_frames = 0
                self.last_print_time = current_time

        except Exception as e:
            self.get_logger().warn(f"⚠️ 读取帧出错: {e}")

    def stop(self):
        """停止摄像头"""
        if hasattr(self, 'camera_backend') and self.camera_backend is not None:
            self.camera_backend.stop()


def main(args=None):
    rclpy.init(args=args)
    node = CameraImagePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
