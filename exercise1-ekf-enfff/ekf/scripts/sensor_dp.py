#!/usr/bin/env python3

import rospy
from ekf.msg import StateVector4D, MeasurementVector2D

import numpy as np

measurement_noise_var = 0.05  # Actual measurement noise variance (unknown to the user)
pub = rospy.Publisher('dp_sensor_measures', MeasurementVector2D, queue_size=10) # Writes to topic `dp_sensor_measures`

def callback(p: StateVector4D):
    # rospy.loginfo(rospy.get_caller_id() + "\nReceived %s from real system", p)

    z_t = np.array([p.x1, p.x2]) + np.sqrt(measurement_noise_var)*np.random.randn()
    msg = MeasurementVector2D(float(z_t[0]), float(z_t[1]))
    rospy.loginfo(f"Sensor is about to send zt1: {msg.zt1} zt2: {msg.zt2}")

    pub.publish(msg) # Writes the noisy measurement to the topic `sensor_measures`


def sensor():
    rospy.init_node('dp_sensor_measures', anonymous=True) # Write to topic `dp_sensor_measures`
    rospy.Subscriber("dp_real_system", StateVector4D, callback) # Read from topic `dp_real_system`

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    try:
        sensor()
    except rospy.ROSInterruptException:
        print("ops something went wrong good luck have fun fixing it")
        pass
