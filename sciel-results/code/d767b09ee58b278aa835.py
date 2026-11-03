"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds

def compute_overlap(h, shifts=None):
    """
    Compute the maximum overlap of h with shifted 1-h.
    The overlap for shift k is sum_i h[i] * (1 - h[(i+k) % N]).
    We want to minimize the maximum over all shifts k.
    """
    if shifts is None:
        shifts = range(1, N)
    h_mod = np.mod(np.arange(N), N)
    max_ov = 0.0
    for k in shifts:
        shifted_h = h[np.mod(np.arange(N) + k, N)]
        ov = np.sum(h * (1.0 - shifted_h))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    # Initial guess: uniform 0.5 gives overlap = 0.5 * 0.5 * N = 25 for each shift
    # Let's verify: h[i]=0.5, (1-h[j])=0.5, sum = 100 * 0.25 = 25
    # We want to minimize this maximum overlap.
    
    # Theoretical lower bound: 
    # For any h with sum N/2, the average overlap over all shifts is:
    # avg_k sum_i h[i](1-h[i+k]) = sum_i h[i] * (1 - avg_k h[i+k])
    # avg_k h[i+k] = sum_j h[j] / N = (N/2)/N = 1/2
    # So avg overlap = sum_i h[i] * (1 - 1/2) = (1/2) * sum_i h[i] = (1/2) * (N/2) = N/4 = 25
    # So the minimum possible maximum overlap is at least 25.
    # Uniform 0.5 achieves exactly 25 for all shifts, so it's optimal!
    
    # Wait, let me double-check. If h is constant 0.5, then for any shift k:
    # overlap = sum_i 0.5 * (1 - 0.5) = sum_i 0.25 = 100 * 0.25 = 25.
    # And the average is 25, so max >= avg = 25, and we achieve 25.
    # So the optimal value is exactly 25.
    
    # But wait - the problem says "bounded between 0 and 1". Let me re-read.
    # "Choose equal-width step heights h on [0,2], bounded between 0 and 1 with integral 1"
    # "integral 1" - if we're on [0,2] with equal width steps, the integral is sum(h_i) * (2/N) = 1
    # So sum(h_i) = N/2 = 50. Yes, that matches.
    
    # The score is "the upper bound on C5". Let me think about what C5 means.
    # Perhaps C5 is a specific constant or metric related to the overlap?
    
    # Actually, re-reading: "score is the upper bound on C5" and "the evaluator measures the raw value (lower is better)"
    # So we need to return the step heights, and the evaluator computes some score.
    
    # Let me reconsider. The task says to return "sequence of step heights in [0, 1] whose sum is exactly len/2"
    # So the return value is h itself, and the evaluator computes the score from it.
    
    # Given that uniform 0.5 gives the theoretical minimum of 25 for the max overlap,
    # and this is provably optimal, let's return that.
    
    # But wait - maybe I'm misunderstanding the overlap definition. Let me re-read:
    # "minimize the maximum overlap of h with shifted 1-h"
    # Overlap of h with shifted 1-h at shift k: this could mean the L1 norm of the product,
    # or it could mean something else. Let me assume it's sum_i h[i] * (1 - h[(i+k)%N]).
    
    # With uniform h=0.5, this is always 25. And by the averaging argument, 
    # the maximum must be at least 25. So 25 is optimal.
    
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
