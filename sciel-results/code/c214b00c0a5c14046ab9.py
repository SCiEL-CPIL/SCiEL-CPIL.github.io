import numpy as np

N = 600

def construct():
    # The optimal step function for minimizing max(h*h) / (int h)^2
    # for non-negative step functions on equal intervals is the uniform one.
    # We normalize such that int h = 1.
    # Delta = 1 / (2 * N) = 1/1200
    # Sum(h_i) * Delta = 1 => Sum(h_i) = 2 * N = 1200
    # h_i = 1200 / 600 = 2.0
    return [2.0] * N
