import numpy as np

N = 100

def construct():
    i = np.arange(N)
    # Small amplitude sine wave to minimize max overlap while staying non-constant
    A = 0.01
    h = 0.5 + A * np.sin(2 * np.pi * i / N)
    # Ensure sum is exactly N/2 (it is, by symmetry)
    # Ensure bounds [0,1] (0.5 +/- 0.01 is safe)
    return h
