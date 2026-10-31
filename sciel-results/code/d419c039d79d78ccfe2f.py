import numpy as np

N = 400

def construct():
    # Create a Hanning window
    # np.hanning(N) creates a symmetric Hanning window of length N
    # Values range from 0 to 1
    x = np.hanning(N)
    # Ensure it's a sequence of heights
    return x
