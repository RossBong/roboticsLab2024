import qi
import rclpy
import argparse
from rclpy.node import Node
import sys
from std_msgs.msg import String
from qi_unipa_msgs.msg import Sonar
import time

class QiUnipa_sensor(Node):
    
    def __init__(self):
        super().__init__('qi_unipa_sensor')
         # Ottieni i parametri
        self.declare_parameter('ip', '192.168.0.161')
        self.declare_parameter('port', 9559)
        ip = self.get_parameter('ip').get_parameter_value().string_value
        port = self.get_parameter('port').get_parameter_value().integer_value
        
        # Connessione sessione
        self.session = self.set_connection(ip, port)
        
        
        self.sonar_pub = self.create_publisher(Sonar, "/sonar", 10)

        self.sonar_service= self.session.service("ALSonar")
        self.memory_service= self.session.service("ALMemory")

        self.timer = self.create_timer(1.0, self.set_sonar)

        

        
       

    def set_connection(self, ip, port):
        session = qi.Session()
        try:
            session.connect(f"tcp://{ip}:{port}")
        except RuntimeError:
            self.get_logger().error(f"Can't connect to Naoqi at ip \"{ip}\" on port {port}.\n"
                                    "Please check your script arguments.")
            sys.exit(1)
        return session
    
    def set_sonar(self):

        self.sonar_service.subscribe("Sonar_app")
        msg=Sonar()
        msg.front_sonar=self.memory_service.getData("Device/SubDeviceList/Platform/Front/Sonar/Sensor/Value")
        msg.back_sonar=self.memory_service.getData("Device/SubDeviceList/Platform/Back/Sonar/Sensor/Value")
        self.sonar_pub.publish(msg)
        self.sonar_service.unsubscribe("Sonar_app")
        
 

def main(args=None):
    rclpy.init(args=args)
 
    node = QiUnipa_sensor()
    
    rclpy.spin(node)
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()