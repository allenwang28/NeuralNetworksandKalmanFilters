import numpy as np

from sim_abstract import Sim_Abstract


class Simple_Polynomial(Sim_Abstract):
    def f(self, x):
        return 0.5*x**2

    def h(self, x):
        return x

    def F(self, x):
        return x

    def H(self, x):
        return x / x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim = Simple_Polynomial(0, 1, 1)
    T = 20
    for t in range(T):
        sim.process_next()

    plt.plot(range(T), sim.get_all_y())
    plt.plot(range(T), sim.get_all_x())
    plt.show()


