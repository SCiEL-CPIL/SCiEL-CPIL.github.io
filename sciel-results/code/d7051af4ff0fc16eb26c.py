"""Optimized program for second autocorrelation inequality task."""
import numpy as np
from scipy.optimize import linprog

N = 50


def _autoconvolution(h):
    """Compute full linear autocorrelation of h."""
    return np.correlate(h, h, mode='full')


def _objective_value(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h * h)
    ac = _autoconvolution(h)
    ac_inf = np.max(ac)
    if ac_inf < 1e-15:
        return 0.0
    return l2_sq / (l1 * ac_inf)


def construct():
    # We want to maximize ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # For nonnegative h, ||h*h||_inf >= h*h at zero-lag = ||h||_2^2
    # So ratio <= 1/||h||_1. With ||h||_1 = 1, ratio <= 1.
    # The delta function achieves exactly 1.
    #
    # However, let's verify with a more careful approach:
    # Try to find if any distribution can beat 1.0.
    # 
    # Actually, for the specific case where we have 50 points on equal intervals,
    # let's try a few candidate distributions and pick the best one.
    
    candidates = []
    
    # Candidate 1: Single spike at center
    h1 = np.zeros(N)
    h1[N // 2] = 1.0
    candidates.append(h1)
    
    # Candidate 2: Two adjacent spikes
    h2 = np.zeros(N)
    h2[N // 2 - 1] = 0.5
    h2[N // 2] = 0.5
    candidates.append(h2)
    
    # Candidate 3: Uniform over center 3 points
    h3 = np.zeros(N)
    h3[N // 2 - 1:N // 2 + 2] = 1.0 / 3
    candidates.append(h3)
    
    # Candidate 4: Uniform over center 5 points
    h4 = np.zeros(N)
    h4[N // 2 - 2:N // 2 + 3] = 1.0 / 5
    candidates.append(h4)
    
    # Candidate 5: Triangular centered at middle
    h5 = np.zeros(N)
    center = N // 2
    for i in range(N):
        d = abs(i - center)
        if d <= 2:
            h5[i] = 3 - d
    h5 /= np.sum(h5)
    candidates.append(h5)
    
    # Candidate 6: Two spikes with gap
    h6 = np.zeros(N)
    h6[N // 2 - 2] = 0.5
    h6[N // 2 + 2] = 0.5
    candidates.append(h6)
    
    # Candidate 7: Broad uniform
    h7 = np.zeros(N)
    h7[:] = 1.0 / N
    candidates.append(h7)
    
    # Evaluate all candidates
    best_h = candidates[0]
    best_val = _objective_value(candidates[0])
    
    for h in candidates[1:]:
        val = _objective_value(h)
        if val > best_val:
            best_val = val
            best_h = h
    
    # Return as list of floats
    return [float(x) for x in best_h]
