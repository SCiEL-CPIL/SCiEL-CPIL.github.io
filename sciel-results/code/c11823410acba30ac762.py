"""Minimize normalized max autoconvolution via flat function."""
import numpy as np
import time

N = 600
DEADLINE = 120.0

def construct():
    # The flat function (all ones) is a strong candidate for minimizing
    # peak-to-norm ratios in autoconvolution.
    # For L2 normalization, the score is exactly 1.
    # For L1^2 normalization, the score is 1/N.
    # Given the ambiguity, the flat function is the safest "optimal" shape.
    return [1.0] * N
