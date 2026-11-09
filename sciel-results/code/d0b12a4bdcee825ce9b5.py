import numpy as np
from scipy.special import roots_laguerre

def construct():
    # Get the 12 roots of the Laguerre polynomial L_12(x)
    roots = roots_laguerre(12)[0]
    # Sort them to be safe
    roots = np.sort(roots)
    return list(roots)
