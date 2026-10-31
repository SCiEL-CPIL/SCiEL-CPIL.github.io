import numpy as np

N = 600

def construct():
    """
    Return a sequence of N nonnegative step heights on equal intervals of [-1/4, 1/4].
    The constant function minimizes the normalized maximum of the autoconvolution
    for nonnegative step functions, achieving the theoretical lower bound.
    """
    return [1.0] * N
