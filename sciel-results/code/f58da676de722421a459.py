import numpy as np

N = 400

def construct():
    """
    Returns the optimal signed step heights.
    
    The objective is to minimize max(|autoconvolution|) / (sum(f))^2.
    Let f be a vector.
    autoconvolution at lag 0 is sum(f_i^2).
    By Cauchy-Schwarz, (sum(f_i))^2 <= N * sum(f_i^2).
    So sum(f_i^2) / (sum(f_i))^2 >= 1/N.
    The maximum absolute value of the autoconvolution is at least the value at lag 0, which is sum(f_i^2).
    Thus, the objective is always >= 1/N.
    The lower bound 1/N is achieved if and only if f_i are all equal (constant vector).
    Therefore, the optimal solution is a constant vector, e.g., all 1s.
    """
    return np.ones(N)
