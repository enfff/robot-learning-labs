#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import Float32

import numpy as np

measurement_noise_var = 0.05  # Actual measurement noise variance (unknown to the user)
pub = rospy.Publisher('sensor_measures', Float32, queue_size=10) # Writes to topic `sensor_measures`

def callback(p: Point):
    rospy.loginfo(rospy.get_caller_id() + "\nReceived %s from real system", p)

    z_t = np.array([p.x, p.y]) + np.sqrt(measurement_noise_var)*np.random.randn()
    rospy.loginfo(f"Sensor is about to send {z_t[0]}")

    pub.publish(z_t[0]) # Writes the noisy measurement to the topic `sensor_measures`


def sensor():
    # Writes to topic `sensor_measures`
    rospy.init_node('sensor_measures', anonymous=True) # Write to topic `sensor_measures`
    # Reads from topic `real_system`
    rospy.Subscriber("real_system", Point, callback) # Read from topic `real_system`

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    try:
        sensor()
    except rospy.ROSInterruptException:
        pass
