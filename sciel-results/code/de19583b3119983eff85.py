import time
import numpy as np

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin before 150s limit
    
    # The optimal solution for minimizing the value 
    # abs(2 * N * max(autocorr) / sum(h)^2) is a constant vector.
    # Proof: By Cauchy-Schwarz, (sum h)^2 <= N * sum(h^2) = N * autocorr[0].
    # So autocorr[0] >= (sum h)^2 / N.
    # max(autocorr) >= autocorr[0].
    # Value >= 2 * N * ((sum h)^2 / N) / (sum h)^2 = 2.
    # Equality holds if and only if all h_i are equal.
    # Thus, any constant vector h = c * ones(N) with c != 0 yields value 2.
    # The score is seed_value / value = 2 / 2 = 1.0.
    
    # We return a constant vector.
    h = np.ones(N)
    
    # Check deadline (not really needed for this simple op, but required by directive)
    if time.time() > deadline:
         return h
         
    return h
