import numpy as np

def construct():
    # The optimal coefficients derived by minimizing the weighted average of Hermite degrees
    # subject to g(0)=0 and a_12=0.
    # a_0 = -3, a_4 = 1, a_8 = 0
    return np.array([-3.0, 1.0, 0.0])
