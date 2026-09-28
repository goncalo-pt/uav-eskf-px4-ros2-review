import rclpy
from rclpy.node import Node

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleLocalPosition
from rclpy.qos import qos_profile_sensor_data

class OffboardControl(Node):

    def __init__(self):
        super().__init__("offboard_control")
     
        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            10
        )

        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            10
        )

        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            10
        )

        self.counter = 0
        self.position = None

        self.waypoints = [
            [0.0, 0.0, -5.0],
            [10.0, 0.0, -5.0],
            [10.0, 10.0, -5.0],
            [10.0, 10.0, -1.0],
        ]
        self.current_waypoint = 0
        self.waypoint_tolerance = 0.2
        self.position_subscription = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.position_callback,
            qos_profile_sensor_data
        )
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timestamp_us(self):
        return self.get_clock().now().nanoseconds // 1000
    def position_callback(self, msg):
        self.position = msg

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        msg.timestamp = self.timestamp_us()

        self.offboard_control_mode_pub.publish(msg)

    def waypoint_reached(self, waypoints):
        if self.position is None:
            return False
        dx = waypoints[0] - self.position.x
        dy = waypoints[1] - self.position.y
        dz = waypoints[2] - self.position.z

        distance = (dx**2 + dy**2 + dz**2)**0.5

        return distance < self.waypoint_tolerance

    def publish_trajectory_setpoint(self):
        msg = TrajectorySetpoint()

        waypoint = self.waypoints[self.current_waypoint]
        msg.position = waypoint

        msg.yaw = 0.0
        msg.timestamp = self.timestamp_us()

        self.trajectory_setpoint_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self.timestamp_us()

        self.vehicle_command_pub.publish(msg)

    def arm(self):
        self.get_logger().info("Sending ARM command")

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
            1.0
        )

    def enter_offboard_mode(self):
        self.get_logger().info("Requesting OFFBOARD mode")

        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
            1.0,
            6.0
        )

    def timer_callback(self):
        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()

        if self.counter == 10:
            self.enter_offboard_mode()
            self.arm()

        if self.counter > 10:
            waypoint = self.waypoints[self.current_waypoint]

            if self.waypoint_reached(waypoint):

                if self.current_waypoint < len(self.waypoints) - 1:
                    self.current_waypoint += 1

                    self.get_logger().info(
                        f"Moving to waypoint {self.current_waypoint}: "
                        f"{self.waypoints[self.current_waypoint]}"
                    )

        self.counter += 1



def main(args=None):
    rclpy.init(args=args)

    node = OffboardControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
