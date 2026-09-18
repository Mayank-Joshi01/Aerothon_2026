import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory


class ServoController(Node):

    def __init__(self):
        super().__init__('servo_controller')

        # Raspberry Pi BCM GPIO number
        self.gpio_pin = 18

        # Use lgpio GPIO backend
        factory = LGPIOFactory()

        # Servo pulse range
        self.servo = Servo(
            self.gpio_pin,
            pin_factory=factory,
            min_pulse_width=0.5 / 1000,
            max_pulse_width=2.5 / 1000
        )

        self.subscription = self.create_subscription(
            Float64,
            '/servo_angle',
            self.angle_callback,
            10
        )

        self.get_logger().info(
            'Servo controller started on GPIO18'
        )

        self.get_logger().info(
            'Publish angles from -90 to +90 degrees'
        )

    def angle_callback(self, msg):

        angle = msg.data

        # Limit angle
        angle = max(-90.0, min(90.0, angle))

        # Convert:
        # -90 degrees -> -1
        #   0 degrees ->  0
        # +90 degrees -> +1

        normalized = angle / 90.0

        self.servo.value = normalized

        self.get_logger().info(
            f'Servo angle: {angle:.1f} degrees'
        )


def main(args=None):

    rclpy.init(args=args)

    node = ServoController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.servo.detach()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()