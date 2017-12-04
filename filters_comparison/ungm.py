import numpy as np

from sim_abstract import Sim_Abstract


"""
    UNGM - Univariate non-stationary growth model

    Contains a high degree of nonlinearity and bi-modality, causing difficulties
    for filters.
"""

class UNGM(Sim_Abstract):
    def f(self, x):
        return 0.5 * x + (25 * x)/(1 + x**2) + 8 * np.cos(1.2 * (self.k - 1)) 

    def h(self, x):
        return x**2 / 20.

    def F(self, x):
        return -50 * x**2 / ((x**2 + 1)**2) + 25/(x**2 + 1) + 0.5

    def H(self, x):
        return x / 10.


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim = UNGM(0, 1, 1)
    T = 10
    for t in range(T):
        sim.process_next()

    plt.plot(range(T), sim.get_all_y())
    plt.plot(range(T), sim.get_all_x())
    plt.show()


