import numpy as np 
from sim_abstract import Sim_Abstract


"""
    Simple linear


    x = x + w_i
    y = y + v_i
"""

class SimpleSim(Sim_Abstract):
    def f(self, x, dt):
        return x + 1

    def h(self, x):
        return 2*x

    def F(self, x, dt):
        return 1

    def H(self, x):
        return 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sim = UNGM(0, 1, 1)
    T = 10
    for t in range(T):
        sim.process_next()

    plt.plot(range(T), sim.get_all_y())
    plt.plot(range(T), sim.get_all_x())
    plt.show()



