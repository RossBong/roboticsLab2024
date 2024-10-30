import qi
import rclpy
import argparse
from rclpy.node import Node
import sys
from std_msgs.msg import Int32
from geometry_msgs.msg import Pose2D
from qi_unipa_msgs.msg import PostureWithSpeed

class QiUnipa_Movement(Node):
    def __init__(self, args):
        super().__init__('qi_unipa_movement')
        self.session = self.set_connection(args)
        self.subscription = self.create_subscription(Int32, "/state", self.set_state, 10)
        self.subscription = self.create_subscription(Int32, "/joint_angles_with_speed", self.set_joint_angles_with_speed, 10)
        self.subscription = self.create_subscription(Pose2D, "/walk", self.set_walking, 10)
        self.subscription = self.create_subscription(PostureWithSpeed, "/posture", self.set_posture, 10)


    def set_connection(self, args):
        session = qi.Session()
        try:
            session.connect("tcp://" + args.ip + ":" + str(args.port))
        except RuntimeError:
            print ("Can't connect to Naoqi at ip \"" + args.ip + "\" on port " + str(args.port) +".\n"
                    "Please check your script arguments. Run with -h option for help.")
            sys.exit(1)
        return session

    def set_state(self, msg):
        state = msg.data
        state_service = self.session.service("ALMotion")
        if state == 0:
            state_service.wakeUp()
        else:
            state_service.rest()

    def set_joint_angles_with_speed(self, msg):
        names = msg.names
        angles = msg.angles
        speed = msg.speed
        joints_service = self.session.service("ALMotion")
        joints_service.angleInterpolationWithSpeed(names, angles, speed)

    def set_walking(self, msg):
        x = msg.x
        y = msg.y
        theta = msg.theta
        walk_service = self.session.service("ALMotion")
        walk_service.moveTo(x,y,theta)

    def set_posture(self, msg):
        pose_name = msg.postureName
        speed = msg.speed
        posture_service = self.session.service("AlRobotPosture")
        posture_service.goToPosture(pose_name,speed)
    


def main(args=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default="127.0.0.1",
                        help="Robot IP address. On robot or Local Naoqi: use '127.0.0.1'.")
    parser.add_argument("--port", type=int, default=9559,
                        help="Naoqi port number")
    
    args = parser.parse_args()

    rclpy.init(args=args)
    node = QiUnipa_Movement(args)
    
    rclpy.spin(node)
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()