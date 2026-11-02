import numpy as np

N = 600

def construct():
    # We suspect the metric is (sum h_i^2) / (max(h) * sum h_i)
    # Scale invariant. Fix max(h) = 1.
    # Minimize f(x) = (1 + (N-1)x^2) / (1 + (N-1)x)
    # where one element is 1 and N-1 elements are x.
    
    # Analytical derivative zero:
    # (N-1)x^2 + 2x - 1 = 0  (approx for large N)
    # x = (-2 + sqrt(4 + 4(N-1))) / (2(N-1)) = (-1 + sqrt(N)) / (N-1)
    
    x_opt = (-1 + np.sqrt(N)) / (N - 1)
    
    # Ensure x is within [0, 1]
    if x_opt < 0:
        x_opt = 0.0
    if x_opt > 1:
        x_opt = 1.0
        
    heights = [x_opt] * N
    heights[0] = 1.0
    
    return heights
