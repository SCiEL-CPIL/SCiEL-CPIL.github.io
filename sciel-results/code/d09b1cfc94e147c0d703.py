import numpy as np

N = 50

def construct():
    # Theoretical maximum for the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf) 
    # for non-negative h is 1.0, achieved by a Delta function (one non-zero entry).
    # Proof sketch:
    # Let h >= 0. Normalize ||h||_1 = 1.
    # We maximize ||h||_2^2 / ||h*h||_inf.
    # We know ||h*h||_inf >= max(h_i^2).
    # Also ||h||_2^2 = sum h_i^2 <= max(h_i) * sum h_i = max(h_i).
    # So Ratio <= max(h_i) / ||h*h||_inf <= max(h_i) / max(h_i^2) = 1/max(h_i).
    # Since max(h_i) <= 1, Ratio <= 1.
    # Equality holds if max(h_i)=1 (which implies h is a delta) and ||h*h||_inf = 1.
    
    h = [0.0] * N
    h[0] = 1.0
    return h
