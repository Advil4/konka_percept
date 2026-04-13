import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat
from rclpy.node import Node
from sensor_msgs.msg import Image


class OrbbecImagePublisher(Node):
    def __init__(self):
        super().__init__('orbbec_rgb_publisher')
        self.topic_name = '/head_camera/color/image_raw'
        self.publish_fps = 30.0
        self.publisher_ = self.create_publisher(Image, self.topic_name, 10)
        self.bridge = CvBridge()

        # 初始化 Orbbec Pipeline
        self.pipeline = Pipeline()
        self.config = Config()

        try:
            # 获取 RGB 流配置
            profile_list = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            if profile_list is None:
                self.get_logger().error("未找到彩色传感器，请检查连接！")
                return

            # 选择一个默认配置 (640x480, RGB888, 30fps)
            try:
                profile = profile_list.get_video_stream_profile(640, 480, OBFormat.RGB888, 30)
            except:
                # 如果找不到精确匹配，取第一个
                profile = profile_list.get_default_video_stream_profile()

            self.config.enable_stream(profile)
            self.pipeline.start(self.config)
            self.get_logger().info(f"奥比中光摄像头已启动，发布到: {self.topic_name}")

        except Exception as e:
            self.get_logger().error(f"摄像头启动失败: {e}")
            return

        # 创建定时器，以指定 FPS 发布
        self.timer = self.create_timer(1.0 / self.publish_fps, self.timer_callback)

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames(100)
            if frames is None:
                return

            color_frame = frames.get_color_frame()
            if color_frame is None:
                return

            # 获取原始数据
            data = np.asanyarray(color_frame.get_data())
            width = color_frame.get_width()
            height = color_frame.get_height()

            # 如果数据大小不等于 width * height * 3，说明是压缩格式 (如 MJPEG)
            if data.size != width * height * 3:
                # 使用 OpenCV 解码 JPEG/MJPEG 数据
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    self.get_logger().warn("解码图像失败")
                    return
            else:
                # 如果是原始 RGB888 数据
                img_rgb = data.reshape((height, width, 3))
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # 封装为 ROS2 消息
            msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_frame"

            self.publisher_.publish(msg)
            self.get_logger().info("已发送一帧图像数据")

        except Exception as e:
            self.get_logger().warn(f"读取帧出错: {e}")

    def stop(self):
        self.pipeline.stop()


def main(args=None):
    rclpy.init(args=args)
    node = OrbbecImagePublisher()
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
