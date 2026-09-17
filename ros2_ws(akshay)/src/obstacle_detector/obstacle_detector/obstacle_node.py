import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64, Bool


class ObstacleDetector(Node):

    def __init__(self):
        super().__init__("obstacle_detector")

        self.obstacle_distance = 1.0

        self.front_angle = math.radians(30)
        self.side_min_angle = math.radians(30)
        self.side_max_angle = math.radians(90)

        self.servo_left_angle = 45.0
        self.servo_right_angle = -45.0
        self.servo_center_angle = 0.0

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        self.front_pub = self.create_publisher(Float64, "/front_distance", 10)
        self.left_pub = self.create_publisher(Float64, "/left_distance", 10)
        self.right_pub = self.create_publisher(Float64, "/right_distance", 10)
        self.obstacle_pub = self.create_publisher(Bool, "/obstacle_detected", 10)
        self.servo_pub = self.create_publisher(Float64, "/servo_angle", 10)

        self.get_logger().info("Obstacle detector started")
        self.get_logger().info("Listening to /scan")

    def get_min_distance(self, scan, min_angle, max_angle):

        distances = []

        for i, distance in enumerate(scan.ranges):

            angle = scan.angle_min + i * scan.angle_increment

            if min_angle <= angle <= max_angle:

                if math.isfinite(distance):

                    if scan.range_min <= distance <= scan.range_max:
                        distances.append(distance)

        if not distances:
            return float("inf")

        return min(distances)

    def scan_callback(self, scan):

        front_distance = self.get_min_distance(
            scan,
            -self.front_angle,
            self.front_angle
        )

        left_distance = self.get_min_distance(
            scan,
            self.side_min_angle,
            self.side_max_angle
        )

        right_distance = self.get_min_distance(
            scan,
            -self.side_max_angle,
            -self.side_min_angle
        )

        front_msg = Float64()
        front_msg.data = front_distance
        self.front_pub.publish(front_msg)

        left_msg = Float64()
        left_msg.data = left_distance
        self.left_pub.publish(left_msg)

        right_msg = Float64()
        right_msg.data = right_distance
        self.right_pub.publish(right_msg)

        obstacle = front_distance < self.obstacle_distance

        obstacle_msg = Bool()
        obstacle_msg.data = obstacle
        self.obstacle_pub.publish(obstacle_msg)

        servo_msg = Float64()

        if obstacle:

            if left_distance > right_distance:
                servo_msg.data = self.servo_left_angle
                direction = "LEFT"
            else:
                servo_msg.data = self.servo_right_angle
                direction = "RIGHT"

        else:

            servo_msg.data = self.servo_center_angle
            direction = "CENTER"

        self.servo_pub.publish(servo_msg)

        self.get_logger().info(
            f"Front: {front_distance:.2f} m | "
            f"Left: {left_distance:.2f} m | "
            f"Right: {right_distance:.2f} m | "
            f"Obstacle: {obstacle} | "
            f"Direction: {direction}"
        )


def main(args=None):

    rclpy.init(args=args)

    node = ObstacleDetector()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
