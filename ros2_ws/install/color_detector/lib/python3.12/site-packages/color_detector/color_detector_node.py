import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

import cv2
import numpy as np


class ColorDetector(Node):

    def __init__(self):
        super().__init__('color_detector')

        # Convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to camera images
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish servo angle
        self.servo_publisher = self.create_publisher(
            Float64,
            '/servo_angle',
            10
        )

        self.get_logger().info('Color detector started')
        self.get_logger().info('RED -> RIGHT (+60 degrees)')
        self.get_logger().info('GREEN -> LEFT (-60 degrees)')

    def image_callback(self, msg):

        try:
            # Convert ROS image to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )

        except Exception as e:
            self.get_logger().error(
                f'Image conversion failed: {e}'
            )
            return

        # Convert BGR image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -------------------------
        # RED COLOR MASK
        # -------------------------

        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])

        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])

        red_mask1 = cv2.inRange(
            hsv,
            red_lower1,
            red_upper1
        )

        red_mask2 = cv2.inRange(
            hsv,
            red_lower2,
            red_upper2
        )

        red_mask = red_mask1 | red_mask2

        # -------------------------
        # GREEN COLOR MASK
        # -------------------------

        green_lower = np.array([35, 80, 80])
        green_upper = np.array([85, 255, 255])

        green_mask = cv2.inRange(
            hsv,
            green_lower,
            green_upper
        )

        # Count detected pixels
        red_pixels = cv2.countNonZero(red_mask)
        green_pixels = cv2.countNonZero(green_mask)

        # Minimum number of pixels required
        # to consider a color detected
        threshold = 500

        # -------------------------
        # SERVO COMMAND
        # -------------------------

        servo_msg = Float64()

        if red_pixels > threshold and red_pixels > green_pixels:

            servo_msg.data = 60.0

            self.servo_publisher.publish(servo_msg)

            self.get_logger().info(
                f'RED detected -> Servo RIGHT (+60°)'
            )

        elif green_pixels > threshold and green_pixels > red_pixels:

            servo_msg.data = -60.0

            self.servo_publisher.publish(servo_msg)

            self.get_logger().info(
                f'GREEN detected -> Servo LEFT (-60°)'
            )


def main(args=None):

    rclpy.init(args=args)

    node = ColorDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()