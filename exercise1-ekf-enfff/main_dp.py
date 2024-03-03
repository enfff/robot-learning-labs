"""
    Robot Learning
    Exercise 1

    Extended Kalman Filter

    Polito A-Y 2023-2024
"""
import pdb

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from ExtendedKalmanFilterDP import ExtendedKalmanFilterDP

# Discretization time step (frequency of measurements)
deltaTime = 0.01

# Initial true state
x0 = np.array([np.pi/3, np.pi/3, 0.5, 0.5])

# Simulation duration in timesteps
simulationSteps = 400
totalSimulationTimeVector = np.arange(0, simulationSteps*deltaTime, deltaTime)

# System dynamics (continuous, non-linear) in state-space representation (https://en.wikipedia.org/wiki/State-space_representation)


def stateSpaceModel(x, t):
    """
        Dynamics may be described as a system of first-order
        differential equations: 
        dx(t)/dt = f(t, x(t))
    """

    g = 9.81
    l1 = 1 # m
    l2 = 1 # m
    m1 = 1 # kg
    m2 = 1 # kg

    # x = [theta1 theta2 theta_dot_1 theta_dot_2]
    # x = [x1       x2      x3          x4]
    # xdot = [x3    x4      bla         bla]

    th1 = x[0]
    th2 = x[1]
    thdot1 = x[2]
    thdot2 = x[3]

    dxdt = np.array([
        x[2],
        x[3],
        (-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(thdot2**2*l2+thdot1**2*l1*np.cos(th1-th2))) / (l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))),
        2*np.sin(th1-th2)*(thdot1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+thdot2**2*l2*np.cos(th1-th2)) / (l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
    ])

    return dxdt

# True solution x(t)
x_t_true = odeint(stateSpaceModel, x0, totalSimulationTimeVector)

"""
    EKF initialization
"""
# Initial state belief distribution (EKF assumes Gaussian distributions)
x_0_mean = np.zeros(shape=(4, 1))  # column-vector
x_0_mean[0] = x0[0] + 3*np.random.randn()
x_0_mean[1] = x0[1] + 3*np.random.randn()
x_0_mean[1] = x0[2] + 3*np.random.randn()
x_0_mean[1] = x0[3] + 3*np.random.randn()

x_0_cov = 10*np.eye(4, 4)  # initial value of the covariance matrix

# # Exercise Default values
# Q=0.00001*np.eye(2,2)
# R = np.array([[0.05]])

# Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
Q = 0.00001*np.eye(4)

# Measurement noise covariance matrix for EKF
R = 0.05*np.eye(2)

# create the extended Kalman filter object
EKF = ExtendedKalmanFilterDP(x_0_mean, x_0_cov, Q, R, deltaTime)


"""
    Simulate process
"""
measurement_noise_var = 0.05  # Actual measurement noise variance (unknown to the user)

for t in range(simulationSteps-1):
    # PREDICT step
    EKF.forwardDynamics()

    # Measurement model
    # print("x_t_true: ", x_t_true[t,0])
    
    z_t = np.zeros(shape=(2, 1))

    z_t[0] = x_t_true[t, 0] + np.sqrt(measurement_noise_var)*np.random.randn()
    z_t[1] = x_t_true[t, 1] + np.sqrt(measurement_noise_var)*np.random.randn()

    # print("z_t")
    # print(z_t)

    # UPDATE step
    EKF.updateEstimate(z_t)

"""
    Plot the true vs. estimated state variables
"""

# Estimates
# EKF.posteriorMeans
# EKF.posteriorCovariances

time = np.arange(0, simulationSteps)*deltaTime
saved_means = np.hstack(EKF.posteriorMeans).T
saved_covs = np.hstack(EKF.posteriorCovariances).T

fig, ax = plt.subplots(4)

# print(str(np.shape(x_t_true)) + " shape of x_t_true")
# print(x_t_true)

ax[0].plot(time, x_t_true[:, 0], linestyle="dashed", label="Real x1")
ax[0].plot(time, saved_means[:, 0], color="red", label="EKF estimation for x1")
ax[0].legend(loc="lower right")

ax[1].plot(time, x_t_true[:, 1], linestyle="dashed", label="Real x2")
ax[1].plot(time, saved_means[:, 1], color="red", label="EKF estimation for x2")
ax[1].legend(loc="lower right")

ax[2].plot(time, x_t_true[:, 2], linestyle="dashed", label="Real x3")
ax[2].plot(time, saved_means[:, 2], color="red", label="EKF estimation for x3")
ax[2].legend(loc="lower right")

ax[3].plot(time, x_t_true[:, 3], linestyle="dashed", label="Real x4")
ax[3].plot(time, saved_means[:, 3], color="red", label="EKF estimation for x4")
ax[3].legend(loc="lower right")

ax[0].set_title("Angle 1 (x1)")
ax[1].set_title("Angle 2 (x2)")
ax[2].set_title("Angular velocity 1 (x3)")
ax[3].set_title("Angular velocity 2 (x4)")

# x-axis labels have been turned off due to lack of space
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])
ax[3].set_xticklabels([])
plt.show()

saved_covs = np.hstack(EKF.posteriorCovariances).T
