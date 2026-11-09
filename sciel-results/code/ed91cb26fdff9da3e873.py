import time
import numpy as np

N = 600

def construct():
    # Return uniform distribution normalized to sum to 1
    # This minimizes sum(h^2)/sum(h) for any scaling, 
    # and is the global minimum if L1-normalized.
    h = np.ones(N) / N
    return [float(x) for x in h]
