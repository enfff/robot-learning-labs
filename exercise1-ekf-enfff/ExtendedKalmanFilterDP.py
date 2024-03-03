"""
    Implementation of the Extended Kalman Filter
    for an unactuated pendulum system
"""
import numpy as np 

class ExtendedKalmanFilterDP(object):
    
    
    def __init__(self, x0, P0, Q, R, dT):
        """
           Initialize EKF
            
            Parameters
            x0 - mean of initial state prior
            P0 - covariance of initial state prior
            Q  - covariance matrix of the process noise 
            R  - covariance matrix of the measurement noise
            dT - discretization step for forward dynamics
        """
        self.x0=x0
        self.P0=P0
        self.Q=Q
        self.R=R
        self.dT=dT
        
        
        self.g = 9.81  # Gravitational constant
        self.l1 = 1  # Length of the pendulum 
        self.l2 = 1
        
        self.currentTimeStep = 0
        
        self.priorMeans = []
        self.priorMeans.append(None)  # no prediction step for timestep=0
        self.posteriorMeans = []
        self.posteriorMeans.append(x0)
        
        self.priorCovariances=[]
        self.priorCovariances.append(None)  # no prediction step for timestep=0
        self.posteriorCovariances=[]
        self.posteriorCovariances.append(P0)
    

    def stateSpaceModel(self, x, t):
        """
            Dynamics may be described as a system of first-order
            differential equations: 
            dx(t)/dt = f(t, x(t))time = np.arange(0, simulationSteps)*deltaTime
            saved_means = np.hstack(EKF.posteriorMeans).T
            saved_covs = np.hstack(EKF.posteriorCovariances).T


            Dynamics are time-invariant in our case, so t is not used.
            
            Parameters:
                x : state variables (column-vector)
                t : time

            Returns:
                f : dx(t)/dt, describes the system of ODEs
        """

        th1 = x[0]
        th2 = x[1]
        thdot1 = x[2]
        thdot2 = x[3]

        l1 = 1 # m
        l2 = 1 # m
        m1 = 1 # kg
        m2 = 1 # kg

        g = self.g

        dxdt = np.array([
            x[2],
            x[3],
            (-g*(2*m1+m2)*np.sin(th1)-m2*g*np.sin(th1-2*th2)-2*np.sin(th1-th2)*m2*(thdot2**2*l2+thdot1**2*l1*np.cos(th1-th2))) / (l1*(2*m1+m2-m2*np.cos(2*th1-2*th2))),
            2*np.sin(th1-th2)*(thdot1**2*l1*(m1+m2)+g*(m1+m2)*np.cos(th1)+thdot2**2*l2*np.cos(th1-th2)) / (l2*(2*m1+m2-m2*np.cos(2*th1-2*th2)))
        ])
        
        return dxdt
    

    def discreteTimeDynamics(self, x_t):
        """
            Forward Euler integration.
            
            returns next state as x_t+1 = x_t + dT * (dx/dt)|_{x_t}
        """
        x_tp1 = x_t + self.dT*self.stateSpaceModel(x_t, None)
        return x_tp1
    

    def jacobianStateEquation(self, x_t):
        """
            Jacobian of discrete dynamics w.r.t. the state variables,
            evaluated at x_t

            Parameters:
                x_t : state variables (column-vector)
            
            Returns
                A   : Jacobian of the discrete system dynamics
        """

        A = np.zeros(shape=(4,4))
        # compute the Jacobian of the discrete dynamics

        x1 = x_t[0]
        x2 = x_t[1]
        x3 = x_t[2]
        x4 = x_t[3]

        # FOR REASONS UNKNOWN TO MANKIND DEFINING A MATRIX LIKE THIS IS NOT ALLOWED
        # A = np.matrix([
        #     [                                                                                                                                                                                                                                                                                        0,                                                                                                                                                                                                                                                     0,                                                       1,                                                        0],
        #     [                                                                                                                                                                                                                                                                                        0,                                                                                                                                                                                                                                                     0,                                                       0,                                                        1],
        #     [                                         (981*np.cos(x1 - 2*x2))/100 + np.cos(x1) + (2*np.cos(x1 - x2)*(np.cos(x1 - x2)*x3**2 + x4**2))/(np.cos(2*x1 - 2*x2) - 3) - (2*x3**2*np.sin(x1 - x2)**2)/(np.cos(2*x1 - 2*x2) - 3) + (4*np.sin(x1 - x2)*np.sin(2*x1 - 2*x2)*(np.cos(x1 - x2)*x3**2 + x4**2))/(np.cos(2*x1 - 2*x2) - 3)**2,                 (2*x3**2*np.sin(x1 - x2)**2)/(np.cos(2*x1 - 2*x2) - 3) - (2*np.cos(x1 - x2)*(np.cos(x1 - x2)*x3**2 + x4**2))/(np.cos(2*x1 - 2*x2) - 3) - (981*np.cos(x1 - 2*x2))/50 - (4*np.sin(x1 - x2)*np.sin(2*x1 - 2*x2)*(np.cos(x1 - x2)*x3**2 + x4**2))/(np.cos(2*x1 - 2*x2) - 3)**2, (4*x3*np.cos(x1 - x2)*np.sin(x1 - x2))/(np.cos(2*x1 - 2*x2) - 3),               (4*x4*np.sin(x1 - x2))/(np.cos(2*x1 - 2*x2) - 3)],
        #     [- (2*np.sin(x1 - x2)*(- np.sin(x1 - x2)*x4**2 + (981*np.sin(x1))/50))/(np.cos(2*x1 - 2*x2) - 3) - (2*np.cos(x1 - x2)*(2*x3**2 + np.cos(x1 - x2)*x4**2 - (981*np.cos(x1))/50))/(np.cos(2*x1 - 2*x2) - 3) - (4*np.sin(x1 - x2)*np.sin(2*x1 - 2*x2)*(2*x3**2 + np.cos(x1 - x2)*x4**2 - (981*np.cos(x1))/50))/(np.cos(2*x1 - 2*x2) - 3)**2, (2*np.cos(x1 - x2)*(2*x3**2 + np.cos(x1 - x2)*x4**2 - (981*np.cos(x1))/50))/(np.cos(2*x1 - 2*x2) - 3) - (2*x4**2*np.sin(x1 - x2)**2)/(np.cos(2*x1 - 2*x2) - 3) + (4*np.sin(x1 - x2)*np.sin(2*x1 - 2*x2)*(2*x3**2 + np.cos(x1 - x2)*x4**2 - (981*np.cos(x1))/50))/(np.cos(2*x1 - 2*x2) - 3)**2,             -(8*x3*np.sin(x1 - x2))/(np.cos(2*x1 - 2*x2) - 3), -(4*x4*np.cos(x1 - x2)*np.sin(x1 - x2))/(np.cos(2*x1 - 2*x2) - 3)]
        # ])

        A[0,0] = 0
        A[0,1] = 0
        A[0,2] = 1
        A[0,3] = 0

        A[1,0] = 0
        A[1,1] = 0
        A[1,2] = 0
        A[1,3] = 1

        A[2,0] = (981*np.cos(x1 - 2*x2))/10000 + np.cos(x1)/100 + (np.cos(x1 - x2)*(np.cos(x1 - x2)*x3**2 + x4**2))/(50*(np.cos(2*x1 - 2*x2) - 3)) - (x3**2*np.sin(x1 - x2)**2)/(50*(np.cos(2*x1 - 2*x2) - 3)) + (np.sin(x1 - x2)*np.sin(2*x1 - 2*x2)*(np.cos(x1 - x2)*x3**2 + x4**2))/(25*(np.cos(2*x1 - 2*x2) - 3)**2)
        A[2,1] = (x3**2*np.sin(x1 - x2)**2)/(50*(np.cos(2*x1 - 2*x2) - 3)) - (np.cos(x1 - x2)*(np.cos(x1 - x2)*x3**2 + x4**2))/(50*(np.cos(2*x1 - 2*x2) - 3)) - (981*np.cos(x1 - 2*x2))/5000 - (np.sin(x1 - x2)*np.sin(2*x1 - 2*x2)*(np.cos(x1 - x2)*x3**2 + x4**2))/(25*(np.cos(2*x1 - 2*x2) - 3)**2)
        A[2,2] = (x3*np.cos(x1 - x2)*np.sin(x1 - x2))/(25*(np.cos(2*x1 - 2*x2) - 3)) + 1
        A[2,3] = (x4*np.sin(x1 - x2))/(25*(np.cos(2*x1 - 2*x2) - 3))

        A[3,0] = - (np.sin(x1 - x2)*(- np.sin(x1 - x2)*x4**2 + (981*np.sin(x1))/50))/(50*(np.cos(2*x1 - 2*x2) - 3)) - (np.cos(x1 - x2)*(2*x3**2 + np.cos(x1 - x2)*x4**2 - (981*np.cos(x1))/50))/(50*(np.cos(2*x1 - 2*x2) - 3)) - (np.sin(x1 - x2)*np.sin(2*x1 - 2*x2)*(2*x3**2 + np.cos(x1 - x2)*x4**2 - (981*np.cos(x1))/50))/(25*(np.cos(2*x1 - 2*x2) - 3)**2)
        A[3,1] = (np.cos(x1 - x2)*(2*x3**2 + np.cos(x1 - x2)*x4**2 - (981*np.cos(x1))/50))/(50*(np.cos(2*x1 - 2*x2) - 3)) - (x4**2*np.sin(x1 - x2)**2)/(50*(np.cos(2*x1 - 2*x2) - 3)) + (np.sin(x1 - x2)*np.sin(2*x1 - 2*x2)*(2*x3**2 + np.cos(x1 - x2)*x4**2 - (981*np.cos(x1))/50))/(25*(np.cos(2*x1 - 2*x2) - 3)**2)
        A[3,2] = -(2*x3*np.sin(x1 - x2))/(25*(np.cos(2*x1 - 2*x2) - 3))
        A[3,3] = 1 - (x4*np.cos(x1 - x2)*np.sin(x1 - x2))/(25*(np.cos(2*x1 - 2*x2) - 3))

        return A
    
    
    def jacobianMeasurementEquation(self, x_t):
        """
            Jacobian of measurement model.

            Measurement model is linear, hence its Jacobian
            does not actually depend on x_t
        """

        C = np.zeros(shape=(2, 4))
        C[0, 0] = 1 # x1 = theta1
        C[0, 1] = 0
        C[1, 0] = 0
        C[1, 2] = 0
        
        C[1, 0] = 0
        C[1, 1] = 1 # x2 = theta2
        C[1, 2] = 0
        C[1, 3] = 0

        return C
    
     
    def forwardDynamics(self):
        self.currentTimeStep = self.currentTimeStep+1  # t-1 ---> t

        
        """
            Predict the new prior mean for timestep t
        """

        x_t_prior_mean = self.discreteTimeDynamics(self.posteriorMeans[self.currentTimeStep-1])
        

        """
            Predict the new prior covariance for timestep t
        """
        # Linearization: jacobian of the dynamics at the current a posteriori estimate

        # print(self.jacobianStateEquation(self.posteriorMeans[self.currentTimeStep-1]))
        A_t_minus = self.jacobianStateEquation(self.posteriorMeans[self.currentTimeStep-1])

        x_t_prior_cov = A_t_minus @ self.posteriorCovariances[-1] @ A_t_minus.T + self.Q

        
        # Save values
        self.priorMeans.append(x_t_prior_mean)
        self.priorCovariances.append(x_t_prior_cov)


    def updateEstimate(self, z_t):
        """
            Compute Posterior Gaussian distribution,
            given the new measurement z_t
        """

        # print("z_t: ", z_t)

        # Jacobian of measurement model at x_t
        Ct = self.jacobianMeasurementEquation(self.priorMeans[self.currentTimeStep]) 
        
        # print("self.priorCovariances shape = " + str(np.shape(self.priorCovariances[-1]))) # 4,4
        # print("self.Ct.T shape = " + str(np.shape(Ct.T))) #4,1
        # print("self.R shape = " + str(np.shape(self.R))) #4,1

        # Compute the Kalman gain matrix
        K_t = self.priorCovariances[-1]@Ct.T@np.linalg.inv(Ct@self.priorCovariances[-1]@Ct.T+self.R)
        
        # Compute posterior mean
        x_t_mean = self.priorMeans[-1] + K_t@(z_t - Ct@self.priorMeans[-1])
        
        # Compute posterior covariance
        x_t_cov = (np.identity(4)-K_t@Ct)@self.priorCovariances[-1]
        
        # Save values
        self.posteriorMeans.append(x_t_mean)
        self.posteriorCovariances.append(x_t_cov)