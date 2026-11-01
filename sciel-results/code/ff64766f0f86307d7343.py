import numpy as np
from scipy.optimize import brentq
import math

def construct():
    # Basis indices
    ks = [0, 4, 8, 12]
    
    # Precompute mu_k = sqrt(pi) * 2^k * k!
    mus = np.zeros(4)
    for i, k in enumerate(ks):
        mus[i] = math.sqrt(math.pi) * (2 ** k) * math.factorial(k)
        
    # h_k = H_k(0)
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 665280
    hs = np.array([1.0, 12.0, 1680.0, 665280.0])
    
    # alpha_k = k + 0.5
    alphas = np.array([k + 0.5 for k in ks])
    
    # w_k = h_k / sqrt(mu_k)
    ws = hs / np.sqrt(mus)
    ws_sq = ws ** 2
    
    # Function to find delta
    def f(delta):
        return np.sum(ws_sq / (alphas - delta))
    
    # Find root of f(delta) = 0
    # We know there is a root in (0.5, 4.5) because f(0.5+) -> -inf and f(4.5-) -> +inf?
    # Let's check signs.
    # For delta > 0.5, term 0 is negative.
    # For delta < 4.5, term 1 is positive.
    # f(0.6) should be very negative.
    # f(4.4) should be very positive.
    # So brentq on [0.51, 4.49]
    
    try:
        delta = brentq(f, 0.5001, 4.4999)
    except Exception:
        # Fallback if brentq fails
        delta = 1.0 
        
    # Compute y_k
    # y_k = -gamma * w_k / (2 * (alpha_k - delta))
    # We can ignore -gamma/2 as it's a common scale factor
    y_unnorm = ws / (alphas - delta)
    
    # Normalize y
    y_norm = y_unnorm / np.linalg.norm(y_unnorm)
    
    # Compute c_k
    c = y_norm / np.sqrt(mus)
    
    # Return c_0, c_4, c_8
    return np.array([c[0], c[1], c[2]])
