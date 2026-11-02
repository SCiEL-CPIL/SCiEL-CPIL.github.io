import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START
def construct():
    """
    Minimize max(|autocorr(x,x)|) / (sum(x))^2 for x in {-1, 1}.
    
    Key insights:
    1. For x in {-1,1}, autocorr at lag 0 is always N (sum of x_i^2).
    2. The denominator (sum(x))^2 is maximized when all x_i are equal (sum = ±N), giving denom = N^2.
    3. In that case, the ratio is N/N^2 = 1/N, which is the theoretical minimum.
    4. However, we need to verify this is achievable and optimal.
    
    Actually, let's think more carefully:
    - If x = all 1s, autocorr at lag 0 = N, at other lags = N - |lag| (for lag < N).
    - So max |autocorr| = N (at lag 0).
    - sum(x) = N, so (sum(x))^2 = N^2.
    - Ratio = N / N^2 = 1/N.
    
    Can we do better? The numerator is at least N (since autocorr at lag 0 = sum(x_i^2) = N for x in {-1,1}).
    The denominator is at most N^2 (when sum(x) = ±N).
    So the ratio is at least N/N^2 = 1/N.
    
    Therefore, the constant sequence achieves the global optimum.
    """
    # The theoretical minimum is achieved by constant sequences.
    # Return all ones (or all -1s, both work).
    return np.ones(N)
# EVOLVE-BLOCK-END
