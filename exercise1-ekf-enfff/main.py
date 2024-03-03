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

from ExtendedKalmanFilter import ExtendedKalmanFilter

# Discretization time step (frequency of measurements)
deltaTime = 0.01

# Initial true state
x0 = np.array([np.pi/3, 0.5])

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
    l = 1
    dxdt = np.array([x[1], -(g/l)*np.sin(x[0])])
    return dxdt



# True solution x(t)
x_t_true = odeint(stateSpaceModel, x0, totalSimulationTimeVector)


"""
    EKF initialization
"""
# Initial state belief distribution (EKF assumes Gaussian distributions)
x_0_mean = np.zeros(shape=(2, 1))  # column-vector
x_0_mean[0] = x0[0] + 3*np.random.randn()
x_0_mean[1] = x0[1] + 3*np.random.randn()
x_0_cov = 10*np.eye(2, 2)  # initial value of the covariance matrix

# # Exercise Default values
# Q=0.00001*np.eye(2,2)
# R = np.array([[0.05]])

# Process noise covariance matrix (close to zero, we do not want to model noisy dynamics)
Q = 0.0001*np.eye(2,2)

# Measurement noise covariance matrix for EKF
R = np.array([[0.05]])

# create the extended Kalman filter object
EKF = ExtendedKalmanFilter(x_0_mean, x_0_cov, Q, R, deltaTime)


"""
    Simulate process
"""
measurement_noise_var = 0.05  # Actual measurement noise variance (unknown to the user)

for t in range(simulationSteps-1):
    # PREDICT step
    EKF.forwardDynamics()

    # Measurement model
    # print("x_t_true: ", x_t_true[t,0])
    z_t = x_t_true[t, 0] + np.sqrt(measurement_noise_var)*np.random.randn()

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

fig, ax = plt.subplots(2)

ax[0].plot(time, x_t_true[:, 0], linestyle="dashed", label="Real x1")
ax[0].plot(time, saved_means[:, 0], color="red", label="EKF estimation for x1")
ax[0].legend(loc="lower right")

ax[1].plot(time, x_t_true[:, 1], linestyle="dashed", label="Real x2")
ax[1].plot(time, saved_means[:, 1], color="red", label="EKF estimation for x2")
ax[1].legend(loc="lower right")

ax[0].set_title("Angle (x1)")
ax[1].set_title("Angular velocity (x2)")

# plt.xlabel(f"Q = {Q[0,0]}*I, R = {R[0,0]}")
plt.show()

saved_covs = np.hstack(EKF.posteriorCovariances).T
