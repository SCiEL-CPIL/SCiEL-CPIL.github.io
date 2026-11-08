import time
import numpy as np

N = 100

def construct():
    """
    Returns a sequence of step heights h in [0, 1] whose sum is exactly N/2.
    The constant function h=0.5 minimizes the maximum circular overlap of h with 1-h.
    """
    return np.full(N, 0.5)
