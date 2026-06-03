from geometry_msgs.msg import Twist


class ROSSender:

    def __init__(self, node):
        # ROS2 publisher — sends velocity commands to the robot on /cmd_vel
        self._pub = node.create_publisher(Twist, "/cmd_vel", 10)

    def send(self, linear, angular):
        msg = Twist()
        msg.linear.x = float(linear)    # forward speed in m/s
        msg.angular.z = float(angular)  # turn speed in rad/s
        self._pub.publish(msg)

    def stop(self):
        self.send(0.0, 0.0)


class FollowController:

    def __init__(self, config, sender):
        self.cfg = config
        self.sender = sender

    def follow(self, distance, angle_rad):
        # how fast to move forward — depending to how far away they are
        if distance is None or distance < self.cfg.min_distance:
            linear = 0.0
        else:
            linear = self.cfg.k_lin * (distance - self.cfg.target_distance)
            linear = min(max(linear, -self.cfg.max_lin_speed), self.cfg.max_lin_speed)

        angular = self.cfg.k_ang * angle_rad
        angular = min(max(angular, -self.cfg.max_ang_speed), self.cfg.max_ang_speed)

        self.sender.send(linear, angular)


    # search behaviour when target lost
    def search(self, last_angle_error):
        if last_angle_error >= 0:
            self.sender.send(0.0, self.cfg.search_turn_speed)
        else:
            self.sender.send(0.0, -self.cfg.search_turn_speed)

    def stop(self):
        self.sender.stop()
