"""
    This file simulates the output of a single pendulum
"""

import rospy

# from ekf.msg import StateVector4D
# Doesn't work I don't know why nor how to solve it, I tried EVERTHING.
# The message itself is correctly shown when running `rosmsg list | grep StateVector4D`
# Edit: I managed to get it working but there's no need to use it anymore

from ekf.msg import StateVector4D

import numpy as np
from scipy.integrate import odeint

# Discretization time step (frequency of measurements)
deltaTime = 0.01
# Initial true state
x0 = np.array([np.pi/3, np.pi/3, 0.5, 0.5])


def stateSpaceModel(x, t=None):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))
    """

    th1 = x[0]
    th2 = x[1]
    thdot1 = x[2]
    thdot2 = x[3]

    l1 = 1 # m
    l2 = 1 # m
    m1 = 1 # kg
    m2 = 1 # kg

    g = 9.81

    dxdt = np.array([
        x[2],
        x[3],
        (-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(thdot2**2*l2+thdot1**2*l1*np.cos(th1-th2))) / (l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))),
        2*np.sin(th1-th2)*(thdot1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+thdot2**2*l2*np.cos(th1-th2)) / (l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    ])
    
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
    pub = rospy.Publisher('dp_real_system', StateVector4D, queue_size=10)
    rospy.init_node('double_pendulum', anonymous=True)
    rate = rospy.Rate(10)  # 10hz

    # Used to set the initial condition
    x_prev = discreteTimeDynamics(x0)

    while not rospy.is_shutdown():

        # Yes I know I might save a variable, but this is much easier to read
        x_next = discreteTimeDynamics(x_prev)
        x_prev = x_next

        msg = StateVector4D(float(x_next[0]), float(x_next[1]), float(x_next[2]), float(x_next[3]))

        # rospy.loginfo(msg)  # Verbose output to show the message
        pub.publish(msg)

        rate.sleep()


if __name__ == '__main__':
    try:
        pendulum()
    except rospy.ROSInterruptException:
        pass
