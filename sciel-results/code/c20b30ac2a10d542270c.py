import numpy as np

N = 600

def construct():
    """
    Returns the optimal step function heights.
    
    The task is to minimize V = max(f*f) / ||f||_1^2 for a nonnegative step function.
    For nonnegative functions, the maximum of the autoconvolution occurs at lag 0.
    The value at lag 0 is proportional to the L2 norm squared (sum of squares of heights).
    Subject to a fixed L1 norm (sum of heights), the sum of squares is minimized 
    when the heights are uniform (by Cauchy-Schwarz inequality).
    
    Therefore, the uniform distribution on the full domain [-1/4, 1/4] is the 
    global minimum for this metric.
    
    We return the uniform heights.
    """
    # Uniform heights
    # Normalized so that sum(heights) = 1. 
    # The metric is scale-invariant, so the absolute scale doesn't matter, 
    # but sum=1 is a standard canonical form.
    v = [1.0 / N] * N
    return v
