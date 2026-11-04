import numpy as np

N = 50

def construct():
    # Create an array of 50 elements
    # Set even indices to 1.0 and odd indices to 0.0
    # This creates 25 separated blocks
    h = np.zeros(N)
    h[::2] = 1.0
    return h
