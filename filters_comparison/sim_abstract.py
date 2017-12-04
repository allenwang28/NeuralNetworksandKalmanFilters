from abc import ABC, abstractmethod
import numpy as np

class Sim_Abstract(ABC):
    """
    Abstract class for simulations.

    Follow the state-space model, i.e.,

    x_i = f_i(x_(i-1)) + w_i
    y_i = h_i(x_i) + v_i

    Assume zero mean Gaussian w_i, v_i with covariances Q, R, respectively
    """
    def __init__(self,
                 x_0,
                 Q,
                 R,
                 dt=1.):
        x_0 = np.array(x_0)
        self.x_0 = x_0
        self.all_x = []
        self.all_y = []
        self.all_w = []
        self.all_v = []

        self.x = x_0

        self.Q = Q
        self.R = R

        self.k = 0
        self.dt = dt

    @abstractmethod
    def f(self, x, dt):
        pass

    @abstractmethod
    def h(self, x):
        pass

    """
    Jacobian function of f
    """
    @abstractmethod
    def F(self, x, dt):
        pass

    """
    Jacobian function of h
    """
    @abstractmethod
    def H(self, x):
        pass

    """
    Process the next step
    """
    def process_next(self):
        self.k += 1
        x = self.f(self.x, self.dt) + np.random.normal(0, self.Q, self.x.shape) 
        y = self.h(x)
        y += np.random.normal(0, self.R, y.shape)
        self.all_x.append(x)
        self.all_y.append(y)
        self.x = x
        return x, y


    """
    Return x, i.e. the true state
    """
    def get_x(self):
        return self.x

    """
    Return y, i.e. the noisy observation
    """
    def get_y(self):
        return self.y

    def get_all_x(self):
        return self.all_x

    def get_all_y(self):
        return self.all_y
