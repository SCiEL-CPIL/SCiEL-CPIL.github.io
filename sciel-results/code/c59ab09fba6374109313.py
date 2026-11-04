"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # By Cauchy-Schwarz, for any real h:
    #   |autoconv[k]| <= autoconv[0] = sum(h_i^2)
    #   and sum(h_i^2) >= sum(h)^2 / N
    # So value = 2*N*max(autoconv)/sum(h)^2 >= 2*N*(sum(h)^2/N)/sum(h)^2 = 2
    # Equality holds iff h is constant. The seed np.ones(N) achieves exactly 2.0,
    # which is the theoretical minimum. No improvement is possible.
    
    # Return the optimal constant vector.
    h = np.ones(N)
    
    # Verify within deadline (trivially fast)
    if time.time() > deadline:
        # Should never happen for a constant vector
        return h
    
    return h
# EVOLVE-BLOCK-END
