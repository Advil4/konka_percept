import time
import os
import glob

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
    CAMERA_TYPE_LOCAL_FOLDER = "local_folder"

    def __init__(self):
        self.camera_type = self.CAMERA_TYPE_LOCAL_FOLDER

        self.topic_name = '/head_camera/color/image_raw'
        self.frame_id = 'camera_color_frame'
        self.publish_fps = 30.0
        self.queue_size = 10

        self.orbbec_width = 640
        self.orbbec_height = 480
        self.orbbec_format = 'RGB888'
        self.orbbec_fps = 30

        self.realsense_width = 640
        self.realsense_height = 480
        self.realsense_format = 'RGB8'
        self.realsense_fps = 30
        self.realsense_enable_auto_exposure = True
        self.realsense_enable_auto_white_balance = True

        self.local_folder_path = '/home/ubuntu/datasets/orin_data/case_20/rgb'
        self.local_folder_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        self.local_folder_loop = True
        self.local_folder_sort_by_name = True


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


class LocalImageFolderBackend:
    """本地图片文件夹后端"""

    def __init__(self, config, logger):
        self.logger = logger
        self.folder_path = config.local_folder_path
        self.loop = config.local_folder_loop
        self.sort_by_name = config.local_folder_sort_by_name

        self.image_paths = []
        self.current_index = 0
        self.total_images = 0

        try:
            if not os.path.exists(self.folder_path):
                raise FileNotFoundError(f"文件夹不存在: {self.folder_path}")

            for ext in config.local_folder_extensions:
                pattern = os.path.join(self.folder_path, ext)
                self.image_paths.extend(glob.glob(pattern))

            pattern_upper = os.path.join(self.folder_path, ext.upper())
            self.image_paths.extend(glob.glob(pattern_upper))

            if not self.image_paths:
                raise FileNotFoundError(f"在 {self.folder_path} 中未找到任何图片文件")

            if self.sort_by_name:
                self.image_paths.sort()

            self.total_images = len(self.image_paths)
            self.current_index = 0

            self.logger.info(
                f"✅ 本地图片文件夹已加载 [{self.total_images} 张图片] "
                f"[路径: {self.folder_path}] "
                f"[循环模式: {'开启' if self.loop else '关闭'}]"
            )

        except Exception as e:
            self.logger.error(f"❌ 本地图片文件夹初始化失败: {e}")
            raise

    def get_frame(self):
        """获取下一帧图像，返回 BGR 格式的 numpy 数组"""
        if self.total_images == 0:
            return None

        if self.current_index >= self.total_images:
            if self.loop:
                self.current_index = 0
                self.logger.info("🔄 图片列表已循环")
            else:
                self.logger.info("⏹️ 所有图片已发布完毕")
                return None

        image_path = self.image_paths[self.current_index]

        try:
            img_bgr = cv2.imread(image_path)

            if img_bgr is None:
                self.logger.warning(f"⚠️ 无法读取图片: {image_path}")
                self.current_index += 1
                return self.get_frame()

            img_bgr = cv2.rotate(img_bgr, cv2.ROTATE_90_CLOCKWISE)

            self.current_index += 1

            return img_bgr

        except Exception as e:
            self.logger.error(f"❌ 读取图片失败 [{image_path}]: {e}")
            self.current_index += 1
            return self.get_frame()

    def stop(self):
        """清理资源"""
        self.logger.info(f"📁 本地图片文件夹后端已停止 [已发布 {self.current_index}/{self.total_images} 张]")


class CameraImagePublisher(Node):
    """通用图像发布器（支持 Orbbec、RealSense 和本地图片文件夹）"""

    def __init__(self):
        super().__init__('camera_rgb_publisher')

        self.config = CameraConfig()
        self.bridge = CvBridge()

        self.publisher_ = self.create_publisher(
            Image,
            self.config.topic_name,
            self.config.queue_size
        )

        self.total_published_frames = 0
        self.interval_frames = 0
        self.last_print_time = time.time()

        self.camera_backend = self._init_camera()

        if self.camera_backend is not None:
            self.timer = self.create_timer(
                1.0 / self.config.publish_fps,
                self.timer_callback
            )
            self.get_logger().info(f"📷 图像发布器已启动，发布到: {self.config.topic_name}")
        else:
            self.get_logger().error("❌ 图像源初始化失败，节点将退出")

    def _init_camera(self):
        """根据配置初始化对应的图像源后端"""
        try:
            if self.config.camera_type == CameraConfig.CAMERA_TYPE_ORBBEC:
                self.get_logger().info("🔄 初始化 Orbbec 摄像头...")
                return OrbbecCameraBackend(self.config, self.get_logger())

            elif self.config.camera_type == CameraConfig.CAMERA_TYPE_REALSENSE:
                self.get_logger().info("🔄 初始化 RealSense 摄像头...")
                return RealSenseCameraBackend(self.config, self.get_logger())

            elif self.config.camera_type == CameraConfig.CAMERA_TYPE_LOCAL_FOLDER:
                self.get_logger().info("🔄 初始化本地图片文件夹...")
                return LocalImageFolderBackend(self.config, self.get_logger())

            else:
                self.get_logger().error(f"❌ 不支持的图像源类型: {self.config.camera_type}")
                return None

        except Exception as e:
            self.get_logger().error(f"❌ 图像源初始化异常: {e}")
            return None

    def timer_callback(self):
        """定时回调：获取帧并发布"""
        try:
            img_bgr = self.camera_backend.get_frame()

            if img_bgr is None:
                if self.config.camera_type == CameraConfig.CAMERA_TYPE_LOCAL_FOLDER:
                    self.get_logger().info("⏹️ 图片发布完成，停止节点")
                    rclpy.shutdown()
                return

            msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.config.frame_id
            self.publisher_.publish(msg)

            self.total_published_frames += 1
            self.interval_frames += 1

            current_time = time.time()
            elapsed_time = current_time - self.last_print_time

            if elapsed_time >= 2.0:
                actual_fps = self.interval_frames / elapsed_time
                self.get_logger().info(
                    f"🌊 [流水线正常] 总计发布: {self.total_published_frames:05d} 帧 | "
                    f"实时输出帧率: {actual_fps:.1f} FPS"
                )

                self.interval_frames = 0
                self.last_print_time = current_time

        except Exception as e:
            self.get_logger().warn(f"⚠️ 读取帧出错: {e}")

    def stop(self):
        """停止图像源"""
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
