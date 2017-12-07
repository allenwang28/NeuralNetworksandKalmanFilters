import numpy as np
"""
EKF modeling the state-space model below:
x_i = f_i(x_(i-1)) + w_i
y_i = h_i(x_i) + v_i

Note that this does not support the input u

w_i, v_i assumed to be Gaussian

"""

def inv(a):
    if not np.isscalar(a) and a.shape[0] == 1:
        a = a[0]
    if np.isscalar(a):
        return 1 / float(a)
    else:
        return np.linalg.inv(a)

class EKF:
    """
    f: 
        function in state space model
    F:
        Jacobian function of f
    h:
        function in state space model
    H:
        Jacobian function of h
    Q:
        Covariance of noise w
    R:
        Covariance of noise v
    x_0:
        Initial prediction 
    P_0:
        Initial estimate covariance
    """
    def __init__(self,
                f, F,
                h, H,
                Q, R,
                x_0, P_0,
                dt=1.):
        self.f = f
        self.F = F

        self.h = h
        self.H = H


        self.Q = np.array(Q)
        self.R = np.array(R)

        self.x_hat = np.array(x_0)
        self.P = np.array(P_0)
        self.dt = dt

        self.predictions = []
        self.Ps = []

    def predict(self):
        F = self.F(self.x_hat, self.dt)
        self.x_hat = self.f(self.x_hat, self.dt)
        self.P = np.dot(np.dot(F, self.P), self.P.T) + self.Q

    def update(self, y):
        if not np.isscalar(y) and y.shape[0] == 1:
            y = y[0]
        H = self.H(self.x_hat)
        e = y - self.h(self.x_hat)
        S = np.dot(np.dot(H, self.P), H.T) + self.R
        K = np.dot(np.dot(self.P, H.T), inv(S))
        self.x_hat = self.x_hat + np.dot(K, e)
        self.P = self.P - np.dot(np.dot(K, H), self.P)
        self.predictions.append(self.x_hat)
        self.Ps.append(np.linalg.norm(self.P))
        return self.x_hat, self.P

    def get_prediction(self):
        return self.x_hat

    def get_all_predictions(self):
        return self.predictions

    def get_all_Ps(self):
        return self.Ps

if __name__ == "__main__":
    from ungm import UNGM
    from simple_polynomial import Simple_Polynomial
    import matplotlib.pyplot as plt
    from scoring import MSE

    x_0 = np.random.normal(0, 1, 1)
    R = 1
    Q = 1

    sim = UNGM(x_0, R, Q)
    ekf = EKF(sim.f, sim.F,
              sim.h, sim.H,
              sim.Q, sim.R,
              x_0, 1)

    T = 100

    for t in range(T):
        x, y = sim.process_next()
        ekf.predict()
        ekf.update(y)

    plt.plot(range(T), ekf.get_all_predictions())
    plt.plot(range(T), sim.get_all_x())
    """
    plt.plot(range(T), ekf.get_all_Ps())
    """
    plt.show()
