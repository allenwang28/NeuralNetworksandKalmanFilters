from scipy.linalg import cholesky
import numpy as np

class UKF:
    def get_sigma_points(self, x, P):
        if np.isscalar(x):
            x = np.asarray([x])

        n = np.size(x)

        if np.isscalar(P):
            P = np.eye(n) * P

        sigmas = np.zeros((2*n+1,n))
        U = cholesky((n + self.kappa) * P)

        sigmas[0] = x
        for k in range(n):
            sigmas[k+1] = np.subtract(x, -U[k])
            sigmas[n+k+1] = np.subtract(x, U[k])
        return sigmas


    def unscented_transform(self, sigma_points, W, noise_cov):
        x = np.dot(W, sigma_points)
        y = sigma_points - x
        P = y.T.dot(np.diag(W)).dot(y) 
        P += noise_cov
        return (x, P)


    def __init__(self,
                 f, F,
                 h, H,
                 Q, R,
                 kappa,
                 x_0, P_0,
                 dt=1.):
        if np.isscalar(x_0):
            x_0 = np.asarray([x_0])
        self.n = np.size(x_0)

        if np.isscalar(P_0):
            P_0 = np.eye(self.n) * P_0
        if np.isscalar(Q):
            Q = np.eye(self.n) * Q 
        if np.isscalar(R):
            y_temp = h(x_0)
            R = np.eye(np.size(y_temp)) * R

        self.f = f
        self.F = F

        self.h = h
        self.H = H


        self.Q = np.array(Q)
        self.R = np.array(R)

        self.x_hat = x_0
        self.P = P_0

        self.predictions = []
        self.Ps = []
        self.kappa = kappa

        self.num_sigmas = 2 * self.n + 1
        self.dt = dt

        self.W = np.full(self.num_sigmas, 0.5/(self.n + self.kappa))
        self.W[0] = self.kappa / (self.n+self.kappa)


    def predict(self):
        sigma_points = self.f(self.get_sigma_points(self.x_hat, self.P), self.dt)
        self.x_hat, self.P = self.unscented_transform(sigma_points, self.W, self.Q)
        self.sigma_points = sigma_points
    
    def update(self, y):
        sigma_points_f = self.sigma_points
        sigma_points_h = self.h(sigma_points_f)
        zp, Pz = self.unscented_transform(sigma_points_h, self.W, self.R)

        Pxz = np.zeros((np.size(self.x_hat), np.size(y)))
        for i in range(self.num_sigmas):
            dx = sigma_points_f[i] - self.x_hat
            dz = sigma_points_h[i] - zp
            Pxz += self.W[i] * np.outer(dx, dz)

        K = np.dot(Pxz, np.linalg.inv(Pz))
        self.x_hat = self.x_hat + np.dot(K, (y - zp))
        self.P = self.P - np.dot(np.dot(K, Pz), K.T)

        self.predictions.append(self.x_hat)
        self.Ps.append(np.linalg.norm(self.P))

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
    x_0 = 0
    R = 10
    Q = 10

    plt.show()

    sim = UNGM(x_0, R, Q)
    ukf = UKF(sim.f, sim.F,
              sim.h, sim.H,
              sim.Q, sim.R,
              4.,
              x_0, 1)
    T = 20
    for t in range(T):
        x, y = sim.process_next()
        ukf.predict()
        ukf.update(y)

    plt.plot(range(T), ukf.get_all_predictions())
    plt.plot(range(T), sim.get_all_x())
    plt.show()





"""
    from filterpy.kalman import UnscentedKalmanFilter
    from filterpy.kalman import JulierSigmaPoints
    sim = UNGM(x_0, R, Q, 1.)
    T = 20

    ukf = UnscentedKalmanFilter(1, 1, 1, sim.h, sim.f, JulierSigmaPoints(1, 0.))
    predictions = []
    for t in range(T):
        x, y = sim.process_next()

        ukf.predict()
        ukf.update(y)
        predictions.append(ukf.x)


    plt.plot(range(T), sim.get_all_x())
    plt.plot(range(T), predictions)
"""
