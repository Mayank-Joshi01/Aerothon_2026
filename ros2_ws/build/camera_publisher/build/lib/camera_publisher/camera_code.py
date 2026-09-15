import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class CameraPublisher(Node):

    def __init__(self):
        super().__init__('camera_publisher')

        # TCP stream coming from rpicam-vid on the Raspberry Pi host
        self.stream_url = 'tcp://127.0.0.1:5000'

        self.cap = cv2.VideoCapture(self.stream_url)

        if not self.cap.isOpened():
            self.get_logger().error(
                'Could not open camera TCP stream'
            )
            raise RuntimeError('Camera stream could not be opened')

        self.bridge = CvBridge()

        # Publish ROS images
        self.publisher = self.create_publisher(
            Image,
            '/camera/image_raw',
            10
        )

        # Capture approximately 30 frames/sec
        self.timer = self.create_timer(
            1.0 / 30.0,
            self.publish_frame
        )

        self.get_logger().info(
            'Camera publisher started'
        )

        self.get_logger().info(
            'Reading IMX500 stream from tcp://127.0.0.1:5000'
        )

        self.get_logger().info(
            'Publishing images on /camera/image_raw'
        )

    def publish_frame(self):

        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warning(
                'Failed to receive camera frame'
            )
            return

        # Convert OpenCV BGR image to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(
            frame,
            encoding='bgr8'
        )

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera'

        self.publisher.publish(msg)

    def destroy_node(self):

        if self.cap.isOpened():
            self.cap.release()

        super().destroy_node()


def main(args=None):

    rclpy.init(args=args)

    node = CameraPublisher()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()