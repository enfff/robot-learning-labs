"""
    This file simulates the output of a single pendulum
"""

import rospy

# from ekf.msg import StateVector4D
# Doesn't work I don't know why nor how to solve it, I tried EVERTHING.
# The message itself is correctly shown when running `rosmsg list | grep StateVector4D`
# Edit: I managed to get it working but there's no need to use it anymore

from geometry_msgs.msg import Point

import numpy as np
from scipy.integrate import odeint

# Discretization time step (frequency of measurements)
deltaTime = 0.01
# Initial true state
x0 = np.array([np.pi/3, 0.5])


def stateSpaceModel(x, t=None):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))
    """
    g = 9.81
    l = 1
    dxdt = np.array([x[1], -(g/l)*np.sin(x[0])])
    return dxdt


def discreteTimeDynamics(x_p):
    """
        Forward Euler integration.

        returns next state as x_t+1 = x_t + dT * (dx/dt)|_{x_t}
    """
    x_next = x_p + deltaTime*stateSpaceModel(x_p)

    return x_next


def pendulum():
    # Writes to topic `real_system`
    pub = rospy.Publisher('real_system', Point, queue_size=10)
    rospy.init_node('pendulum', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    # Used to set the initial condition
    x_prev = discreteTimeDynamics(x0)

    while not rospy.is_shutdown():

        # Yes I know I might save a variable, but this is much easier to read
        x_next = discreteTimeDynamics(x_prev)
        x_prev = x_next

        msg = Point(x_next[0], x_next[1], 0)

        rospy.loginfo(msg)  # Verbose output to show the message
        pub.publish(msg)

        rate.sleep()


if __name__ == '__main__':
    try:
        pendulum()
    except rospy.ROSInterruptException:
        pass
