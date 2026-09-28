import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from px4_msgs.msg import SensorCombined
from px4_msgs.msg import SensorGps
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleLocalPosition

import time

class PX4ESKF(Node):

    def __init__(self):
        super().__init__("px4_eskf")

        self.prediction_count = 0
        self.prediction_time_sum = 0.0

        self.log_file = open("eskf_vs_px4.csv", "w")
        self.get_logger().info(
            "Opened eskf_vs_px4.csv for logging"
        )
        self.log_file.write(
            "time,"
            "gps_N,gps_E,gps_D,"
            "bg_x,bg_y,bg_z,"
            "ba_x,ba_y,ba_z,"
            "eskf_N,eskf_E,eskf_D,"
            "eskf_vN,eskf_vE,eskf_vD,"
            "px4_N,px4_E,px4_D,"
            "px4_vN,px4_vE,px4_vD\n"
        )

        self.px4_origin = None
        self.log_start_time = None
        
        self.attitude_subscription = self.create_subscription(
            VehicleAttitude,
            "/fmu/out/vehicle_attitude",
            self.attitude_callback,
            qos_profile_sensor_data
        )

        self.q_px4 = None
        self.gps_origin = None
        self.log_timer = self.create_timer(
            0.05,
            self.log_estimates
        )        

        self.debug_timer = self.create_timer(0.5,
                                             self.print_debug)

        self.imu_subscription = self.create_subscription(
            SensorCombined,
            "/fmu/out/sensor_combined",
            self.imu_callback,
            qos_profile_sensor_data
        )

        self.gps_subscription = self.create_subscription(
            SensorGps,
            "/fmu/out/vehicle_gps_position",
            self.gps_callback,
            qos_profile_sensor_data
        )        

        self.omega_m = None
        self.accel_m = None
        self.dt = None

        self.latitude = None
        self.longitude = None
        self.altitude = None
        self.gps_velocity = None
        self.gps_fix_type = None
        self.gps_position_ned = None
        self.gps_velocity_ned = None

        self.g = np.array([0, 0, 9.80665])

        self.px4_position = None
        self.px4_velocity = None
        self.px4_timestamp = None

        # Initial nominal state
        self.x_nom = {"p":(0,0,0),
                      "v":(0,0,0),
                      "R":np.eye(3),
                      "bg":(0,0,0),
                      "ba":(0,0,0)}

        # Initial mean of 
        # belief over error-state
        self.dx = {"p":(0,0,0),
                   "v":(0,0,0),
                   "theta":(0,0,0),
                   "bg":(0,0,0),
                   "ba":(0,0,0)}

        # Covariance matrix of 
        # belief over error-state
        self.P = 0.01*np.eye(15)

        # Covariance for 
        # process noise.
        self.Q = 1e-6*np.eye(15)

        # Gps measurement noise
        # covariance matrix.
        self.R_gps = np.diag([1.0, 1.0, 4.0])

        self.R_gps_complete = np.diag([1.0, 1.0, 4.0, 0.01, 0.01, 0.04])


        I3 = np.eye(3)
        a_noise_cov = (0.02**2) * I3
        w_noise_cov = (0.002**2) * I3
        a_bias_cov = (5e-4**2) * I3
        g_bias_cov = (5e-5**2) * I3
        Z3 = np.zeros((3, 3))
        self.Q_noise = np.block([
            [a_noise_cov, Z3,          Z3,         Z3],
            [Z3,          w_noise_cov, Z3,         Z3],
            [Z3,          Z3,          g_bias_cov, Z3],
            [Z3,          Z3,          Z3,         a_bias_cov]
        ])        

        self.px4_position_subscription = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.px4_position_callback,
            qos_profile_sensor_data
        )    

        self.attitude_initialized = False
        self.gps_initialized = False

    def px4_position_callback(self, msg):

        self.px4_position = np.array([
            msg.x,
            msg.y,
            msg.z
        ])

        self.px4_velocity = np.array([
            msg.vx,
            msg.vy,
            msg.vz
        ])

        self.px4_timestamp = msg.timestamp_sample

    def quat_to_R(self, q):

        q = np.asarray(q, dtype=float)

        # Normalize for numerical safety
        q = q / np.linalg.norm(q)

        w, x, y, z = q

        R = np.array([
            [
                1.0 - 2.0*(y*y + z*z),
                2.0*(x*y - w*z),
                2.0*(x*z + w*y)
            ],
            [
                2.0*(x*y + w*z),
                1.0 - 2.0*(x*x + z*z),
                2.0*(y*z - w*x)
            ],
            [
                2.0*(x*z - w*y),
                2.0*(y*z + w*x),
                1.0 - 2.0*(x*x + y*y)
            ]
        ])

        return R        

    def attitude_callback(self, msg):

        if not self.attitude_initialized:
            q = np.array(msg.q, dtype=float)
            self.x_nom["R"] = self.quat_to_R(q)
            self.attitude_initialized = True



    # Exponential map on SO(3)
    def Exp(self, phi):
        theta = np.linalg.norm(phi)
        if np.isclose(theta, 0.0):
            U = np.zeros((3,3))
        else:
            u = phi / theta
            U = np.array([[0, -u[2], u[1]],
                        [u[2], 0, -u[0]],
                        [-u[1], u[0], 0]])

        I = np.eye(3)

        R = I + np.sin(theta)*U + (1-np.cos(theta))*(U@U)
        return R    
    
    def imu_callback(self, msg):

        self.omega_m = np.array(
            msg.gyro_rad,
            dtype=float
        )

        self.accel_m = np.array(
            msg.accelerometer_m_s2,
            dtype=float
        )
        
        self.dt = msg.gyro_integral_dt * 1e-6
        self.timestamp_imu = msg.timestamp

        if not self.gps_initialized or not self.attitude_initialized:
            return
        start = time.perf_counter()
        self.predictor()
        elapsed = time.perf_counter() - start
        self.prediction_count += 1
        self.prediction_time_sum += elapsed

        if self.prediction_count % 200 == 0:

            mean = (
                self.prediction_time_sum /
                self.prediction_count
            )

            self.get_logger().info(
                f"Predictor mean: {mean*1000:.3f} ms, "
                f"IMU dt: {self.dt*1000:.3f} ms"
            )        
                                          
    # hat operator takes 3-vector 
    # to element of so(3) algebra.
    def hat(self, phi):
        U = np.array([[0, -phi[2], phi[1]],
                      [phi[2], 0, -phi[0]],
                      [-phi[1], phi[0], 0]])   

        return U

    def compute_F(self):
        R = self.x_nom["R"]
        ba = self.x_nom["ba"]
        bg = self.x_nom["bg"]
        f = self.hat(self.accel_m - ba)
        w = self.omega_m - bg
        Z3 = np.zeros((3,3))
        I3 = np.eye(3)
        F23 = -R@f
        F25 = -R
        F33 = -self.hat(w)


        F = np.block([[Z3, I3, Z3, Z3, Z3],
                      [Z3, Z3, F23, Z3, F25],
                      [Z3, Z3, F33, -I3, Z3],
                      [Z3, Z3, Z3, Z3, Z3],
                      [Z3, Z3, Z3, Z3 , Z3]])

        return F


    def predictor(self):

        ba = self.x_nom["ba"]
        bg = self.x_nom["bg"]
        a_meas = self.accel_m
        w_meas = self.omega_m

        R_last = self.x_nom["R"]
        g = self.g
        dt = self.dt
        v_last = np.asarray(self.x_nom["v"])
        p_last = np.asarray(self.x_nom["p"])

        # Subtract nominal biases
        f = a_meas - ba
        w = w_meas - bg

        # Measured acceleration in world frame
        a_ = R_last @ f + g

        # Predicted attitude
        R_ = R_last @ self.Exp(dt* w)

        # Predicted velocity in world frame
        v_ = dt * a_ + v_last

        # Predicted position in world frame
        p_ = p_last + v_last*dt + 0.5*a_*(dt**2)

        self.x_nom["R"] = R_
        self.x_nom["v"] = v_
        self.x_nom["p"] = p_

        # Update covariance matrix 
        # of error-state belief
        
        P = self.P.copy()
        F = self.compute_F()

        Z3 = np.zeros((3,3))
        I3 = np.eye(3)
        R_nom = self.x_nom["R"]
        dt = self.dt
        G = np.block([[Z3, Z3, Z3, Z3],
                    [-R_nom, Z3, Z3, Z3],
                    [Z3, -I3, Z3, Z3],
                    [Z3, Z3, I3, Z3],
                    [Z3, Z3, Z3, I3]]) 
        Q_noise = self.Q_noise
        Qd = (dt**2)*(G @ Q_noise @ G.T)

        # Update covariance of belief over error-state 
        Phi = np.eye(15) + dt * F
        self.P = Phi @ P @ Phi.T + Qd
        


    def gps_callback(self, msg):
        
        self.timestamp_gps = msg.timestamp

        latitude = msg.latitude_deg
        longitude = msg.longitude_deg
        altitude = msg.altitude_msl_m

        self.gps_fix_type = msg.fix_type

        # Store first GPS measurement as local reference
        if self.gps_origin is None:
            if msg.fix_type >= 3:
                self.gps_initialized = True
                self.gps_origin = np.array([
                    latitude,
                    longitude,
                    altitude
                ])

                self.get_logger().info(
                    f"GPS origin initialized: {self.gps_origin}"
                )

            else:
                self.get_logger().warn(
                    "Waiting for valid GPS fix before setting origin"
                )
                return 

        self.gps_position_ned = self.gps_to_ned(
            latitude,
            longitude,
            altitude
        )

        self.gps_velocity_ned = np.array([
            msg.vel_n_m_s,
            msg.vel_e_m_s,
            msg.vel_d_m_s
        ])            
    
        if not self.gps_initialized or not self.attitude_initialized:
            return
        self.update()

    def h_gps(self, x):
        return np.asarray(x["p"])
    
    def h_gps_complete(self, x):
        return np.concatenate((np.asarray(x["p"]),
                               np.asarray(x["v"])))

    # Measurement Jacobian for position
    #  plus velocity gps measurement.
    def compute_H_complete(self):
        I3 = np.eye(3)
        Z3 = np.zeros((3,3))
        H = np.block([[I3,Z3,Z3,Z3,Z3],
                      [Z3,I3,Z3,Z3,Z3]])        

        return H


    def update(self):

        #H = self.compute_H()
        H = self.compute_H_complete()
        P = self.P
        #R_gps = self.R_gps
        R_gps = self.R_gps_complete
        
        S = np.linalg.inv(H @ P @ H.T + R_gps)

        # GPS-measured position
        #z = self.gps_position_ned
        z = np.concatenate((self.gps_position_ned,
                            self.gps_velocity_ned))

        # Nominal state
        x_nom = self.x_nom

        # Kalman gain
        K = P @ H.T @ S

        # Residual
        #r = z - self.h_gps(x_nom)
        r = z - self.h_gps_complete(x_nom)

        # Kalman correction
        dx = K @ r

        # Error injection
        p_new = np.asarray(self.x_nom["p"]) + np.array(dx[0:3])
        v_new = np.asarray(self.x_nom["v"]) + np.array(dx[3:6])
        bg_new = np.asarray(self.x_nom["bg"]) + np.array(dx[9:12])
        ba_new = np.asarray(self.x_nom["ba"]) + np.array(dx[12:])

        dtheta = dx[6:9]
        R = self.x_nom["R"]
        R_new = R @ self.Exp(dtheta)

        # Updated nominal state
        self.x_nom = {"p":p_new,
                      "v":v_new,
                      "R":R_new,
                      "bg":bg_new,
                      "ba":ba_new} 

        # Update belief over 
        # error-state covariance matrix
        P = self.P.copy()
        I_15 = np.eye(15)
        P = (I_15 - K @ H) @ P @ (I_15 - K @ H).T + K @ R_gps @ K.T

        # Reset covariance coordinates
        dtheta = np.asarray(dx[6:9])

        Gamma_theta = np.eye(3) - 0.5 * self.hat(dtheta)

        Gamma = np.eye(15)
        Gamma[6:9, 6:9] = Gamma_theta

        self.P = Gamma @ P @ Gamma.T        
        


    # H matrix for position-only GPS
    def compute_H(self):
        I3 = np.eye(3)
        Z3 = np.zeros((3,3))
        H = np.block([I3,Z3,Z3,Z3,Z3])

        return H

    def gps_to_ned(self, latitude, longitude, altitude):

        lat0, lon0, alt0 = self.gps_origin

        # Convert degrees to radians
        lat = np.deg2rad(latitude)
        lon = np.deg2rad(longitude)

        lat0 = np.deg2rad(lat0)
        lon0 = np.deg2rad(lon0)

        dlat = lat - lat0
        dlon = lon - lon0

        # WSG-84 constants
        a = 6378137.0
        e2 = 6.69437999014e-3

        sin_lat0 = np.sin(lat0)

        # Radius of curvature in meridian
        M = (
            a * (1.0 - e2)
            / (1.0 - e2 * sin_lat0**2)**1.5
        )

        # Radius of curvature in prime vertical
        N_radius = (
            a
            / np.sqrt(1.0 - e2 * sin_lat0**2)
        )

        north = (M + alt0) * dlat

        east = (
            (N_radius + alt0)
            * np.cos(lat0)
            * dlon
        )

        down = alt0 - altitude

        return np.array([
                north,
                east,
                down
        ])


    def print_debug(self):
        if self.omega_m is not None:
            self.get_logger().info(
                "\n"
                f"IMU\n"
                f"  gyro [rad/s]: {self.omega_m}\n"
                f"  accel [m/s^2]: {self.accel_m}\n"
                f"  dt [s]: {self.dt}"
            )

        if self.gps_position_ned is not None:
            self.get_logger().info(
                "\n"
                f"GPS\n"
                f"  N[m]: {self.gps_position_ned[0]}\n"
                f"  E[m]: {self.gps_position_ned[1]}\n"
                f"  D[m]: {self.gps_position_ned[2]}\n"
                f"  fix type: {self.gps_fix_type}"
            )

    def log_estimates(self):

        if self.px4_position is None:
            return

        if not self.gps_initialized or not self.attitude_initialized:
            return

        if self.px4_origin is None:
            self.px4_origin = self.px4_position.copy()

        gps_p = self.gps_position_ned

        bg_eskf = np.asarray(self.x_nom["bg"])
        ba_eskf = np.asarray(self.x_nom["ba"])

        p_eskf = np.asarray(self.x_nom["p"])
        v_eskf = np.asarray(self.x_nom["v"])

        p_px4 = self.px4_position - self.px4_origin
        v_px4 = self.px4_velocity

        # Use PX4 timestamp as logging time
        t = self.px4_timestamp * 1e-6

        if self.log_start_time is None:
            self.log_start_time = t

        t = t - self.log_start_time

        self.log_file.write(
            f"{t},"
            f"{gps_p[0]},{gps_p[1]},{gps_p[2]},"
            f"{bg_eskf[0]},{bg_eskf[1]},{bg_eskf[2]},"
            f"{ba_eskf[0]},{ba_eskf[1]},{ba_eskf[2]},"
            f"{p_eskf[0]},{p_eskf[1]},{p_eskf[2]},"
            f"{v_eskf[0]},{v_eskf[1]},{v_eskf[2]},"
            f"{p_px4[0]},{p_px4[1]},{p_px4[2]},"
            f"{v_px4[0]},{v_px4[1]},{v_px4[2]}\n"
        )

        self.log_file.flush()
def main(args=None):
    rclpy.init(args=args)

    node = PX4ESKF()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.log_file.close()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

