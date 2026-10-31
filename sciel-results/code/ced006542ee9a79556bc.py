import numpy as np

N = 600

def construct():
    # Intervals on [-1/4, 1/4]
    # Length of domain is 0.5
    # Interval width
    dx = 0.5 / N
    # Midpoints
    x = np.linspace(-0.25 + dx/2, 0.25 - dx/2, N)
    
    # Exponential decay
    # tau controls the width. Ratio is independent of tau for pure exponential.
    # Choose a small tau to ensure it's not "constant-like" due to floating point or grid resolution issues
    # though the formula holds for any tau.
    tau = 0.01
    f = np.exp(-np.abs(x) / tau)
    
    # Return as list of floats
    return f.tolist()
