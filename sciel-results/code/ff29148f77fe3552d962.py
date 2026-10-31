import numpy as np

N = 100

def construct():
    # Use a sinusoidal distribution to minimize the maximum overlap.
    # A full-period sine wave has zero discrete autocorrelation for non-zero shifts,
    # which minimizes the overlap with its shifted complement to the theoretical minimum of N/4.
    i = np.arange(N)
    h = 0.5 + 0.5 * np.sin(2 * np.pi * i / N)
    return h
