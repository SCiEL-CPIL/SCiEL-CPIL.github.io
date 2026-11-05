import numpy as np

N = 400

def construct():
    # The seed (all ones) achieves the theoretical minimum value of 2.0
    # by Cauchy-Schwarz: value >= 2*N*sum(h^2)/sum(h)^2 >= 2*N*(sum(h)^2/N)/sum(h)^2 = 2
    # Equality holds iff h is constant, so np.ones(N) is globally optimal.
    return np.ones(N)
