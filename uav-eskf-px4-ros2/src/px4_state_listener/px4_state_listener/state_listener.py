import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import VehicleLocalPosition
from px4_msgs.msg import VehicleAttitude


class PX4StateListener(Node):

    def __init__(self):
        super().__init__("px4_state_listener")

        self.position = None
        self.attitude = None

        # PX4 -> ROS 2 subscriptions need a QoS profile
        # compatible with PX4's publishers.
        self.position_subscription = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            qos_profile_sensor_data
        )

        self.attitude_subscription = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.attitude_callback,
            qos_profile_sensor_data
        )

        # Print the latest state at 2 Hz rather than printing
        # every incoming PX4 message.
        self.timer = self.create_timer(
            0.5,
            self.print_state
        )

    def position_callback(self, msg):
        self.position = msg

    def attitude_callback(self, msg):
        self.attitude = msg

    def print_state(self):

        if self.position is None or self.attitude is None:
            self.get_logger().info("Waiting for PX4 state...")
            return

        p = [
            self.position.x,
            self.position.y,
            self.position.z
        ]

        v = [
            self.position.vx,
            self.position.vy,
            self.position.vz
        ]

        q = self.attitude.q

        self.get_logger().info(
            "\n"
            f"Position NED [m]:       {p}\n"
            f"Velocity NED [m/s]:    {v}\n"
            f"Attitude q [w,x,y,z]:  {list(q)}"
        )


def main(args=None):

    rclpy.init(args=args)

    node = PX4StateListener()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
