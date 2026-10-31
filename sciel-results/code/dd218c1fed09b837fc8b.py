import numpy as np

N = 600

def construct():
    # Create a zero array
    f = np.zeros(N)
    # Set the center to 1.0
    # This minimizes L1 norm relative to L2 norm, 
    # which likely reduces the upper bound if it's L1-based.
    f[N // 2] = 1.0
    return f.tolist()
