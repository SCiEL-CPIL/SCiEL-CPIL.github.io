"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize

N = 100

def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))

def _fix_sum(h, target=None):
    """Adjust h to have exact sum = target (default N/2) while keeping values in [0,1]."""
    if target is None:
        target = N / 2.0
    h = np.clip(h, 0.0, 1.0)
    s = h.sum()
    if abs(s - target) < 1e-14:
        return h
    # Iterative projection
    for _ in range(100):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        if diff > 0:
            mask = h < 1.0 - 1e-12
        else:
            mask = h > 1e-12
        if not mask.any():
            break
        step = diff / mask.sum()
        h[mask] = np.clip(h[mask] + step, 0.0, 1.0)
    return h

def _max_overlap_all_shifts(h):
    """Compute the maximum overlap of h with shifted (1-h) over all cyclic shifts."""
    one_minus_h = 1.0 - h
    max_ov = 0.0
    for s in range(N):
        shifted = np.roll(one_minus_h, s)
        ov = np.dot(h, shifted)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _objective_slsqp(h, alpha=50.0):
    """
    Objective function for SLSQP optimization.
    Uses soft-max of all cyclic shift overlaps.
    """
    one_minus_h = 1.0 - h
    overlaps = np.array([np.dot(h, np.roll(one_minus_h, s)) for s in range(N)])
    m = np.max(overlaps)
    exp_terms = np.exp(alpha * (overlaps - m))
    soft_max = m + np.log(np.sum(exp_terms)) / alpha
    return soft_max

def _gradient_slsqp(h, alpha=50.0):
    """
    Gradient of the soft-max objective.
    """
    one_minus_h = 1.0 - h
    overlaps = np.array([np.dot(h, np.roll(one_minus_h, s)) for s in range(N)])
    m = np.max(overlaps)
    exp_terms = np.exp(alpha * (overlaps - m))
    w = exp_terms / np.sum(exp_terms)
    
    grad = np.zeros(N)
    for s in range(N):
        if w[s] == 0:
            continue
        shifted_omh = np.roll(one_minus_h, s)
        # d/dh_j of dot(h, shifted_omh) = shifted_omh_j
        # But shifted_omh depends on h too!
        # Actually, overlap_s = sum_i h_i * (1 - h_{i-s})
        # d(overlap_s)/d(h_j) = (1 - h_{j-s}) - h_j * delta_{j, i+s}... 
        # Let's use the property: overlap_s = sum_i h_i (1 - h_{i-s})
        # d/dh_j = (1 - h_{j-s}) - sum_i h_i * d(1-h_{i-s})/dh_j
        # d(1-h_{i-s})/dh_j = -1 if i-s == j => i = j+s
        # So d(overlap_s)/dh_j = (1 - h_{j-s}) - h_{j+s} (with indices mod N)
        
        # Wait, the shift is np.roll(one_minus_h, s). 
        # (1-h)_{i} -> (1-h)_{i-s} in the rolled array at position i?
        # np.roll(a, s) moves elements to the right. a[i] goes to a[(i+s)%N].
        # So shifted[i] = one_minus_h[(i-s)%N].
        # overlap_s = sum_i h_i * one_minus_h[(i-s)%N]
        # d(overlap_s)/dh_j = one_minus_h[(j-s)%N] - h_j * 0 ... no.
        # The term is h_j * one_minus_h[(j-s)%N]. Derivative wrt h_j is one_minus_h[(j-s)%N].
        # BUT one_minus_h also depends on h.
        # one_minus_h[k] = 1 - h[k].
        # So overlap_s = sum_i h_i (1 - h[(i-s)%N]).
        # Let k = (i-s)%N => i = (k+s)%N.
        # overlap_s = sum_k h[(k+s)%N] (1 - h[k]).
        # d/dh_j:
        # Term 1: derivative of h[(k+s)%N] wrt h_j is 1 if (k+s)%N == j => k = (j-s)%N.
        #   Contribution: (1 - h[(j-s)%N])
        # Term 2: derivative of (1 - h[k]) wrt h_j is -1 if k == j.
        #   Contribution: -h[(j+s)%N]
        
        idx1 = (j - s) % N
        idx2 = (j + s) % N
        # We need to sum this over all j for each s, weighted by w[s]
        # This is O(N^2) which is fine for N=100.
        
        for j in range(N):
            grad[j] += w[s] * ((1.0 - h[idx1]) - h[idx2])
            
    return grad

def _optimize_slsqp(h0, max_iter=1000, alpha=50.0):
    """
    Optimize using SLSQP with analytical gradient.
    """
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0}]
    bounds = [(0.0, 1.0)] * N
    
    def obj(h):
        return _objective_slsqp(h, alpha)
    
    def grad(h):
        return _gradient_slsqp(h, alpha)
    
    try:
        res = minimize(
            obj, h0, jac=grad, method='SLSQP', bounds=bounds, 
            constraints=constraints, options={'maxiter': max_iter, 'ftol': 1e-8}
        )
        h_opt = _fix_sum(res.x)
        return h_opt, _score(h_opt)
    except Exception:
        return h0, _score(h0)

def _generate_sigmoid_guesses():
    """
    Generate initial guesses based on sigmoid transitions.
    """
    guesses = []
    i = np.arange(N)
    
    # Try different widths and centers
    for w in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        for c in np.linspace(0, N, 5):
            h = 1.0 / (1.0 + np.exp(-(i - c) / w))
            h = _fix_sum(h)
            guesses.append(h)
            
    # Try anti-sigmoid (1 - sigmoid)
    for w in [0.5, 1.0, 2.0, 5.0]:
        for c in np.linspace(0, N, 5):
            h = 1.0 - (1.0 / (1.0 + np.exp(-(i - c) / w)))
            h = _fix_sum(h)
            guesses.append(h)
            
    return guesses

def _local_search_polish(h, max_iter=2000):
    """Coordinate descent local search."""
    h = _fix_sum(h.copy())
    current_score = _score(h)
    
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    
    for step in step_sizes:
        improved = True
        while improved:
            improved = False
            for i in range(N):
                for delta in [step, -step]:
                    h_test = h.copy()
                    h_test[i] = np.clip(h_test[i] + delta, 0.0, 1.0)
                    h_test = _fix_sum(h_test)
                    new_score = _score(h_test)
                    if new_score < current_score - 1e-12:
                        h = h_test
                        current_score = new_score
                        improved = True
                        break
        if not improved:
            break
    
    return h, current_score

def construct():
    deadline = time.time() + 140.0
    
    # Phase 1: Generate diverse initial guesses
    guesses = _generate_sigmoid_guesses()
    
    # Add some random guesses
    for _ in range(10):
        h = np.random.uniform(0.0, 1.0, size=N)
        guesses.append(_fix_sum(h))
        
    # Add the 2-block analytic solution
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h2 = np.zeros(N)
    half = N // 2
    h2[:half] = A
    h2[half:] = B
    guesses.append(_fix_sum(h2))
    
    h_best = None
    score_best = float('inf')
    for h in guesses:
        s = _score(h)
        if s < score_best:
            score_best = s
            h_best = h.copy()
            
    # Phase 2: SLSQP optimization from best guess
    if time.time() < deadline - 20.0:
        h_opt, score_opt = _optimize_slsqp(h_best, max_iter=500, alpha=50.0)
        if score_opt < score_best:
            score_best = score_opt
            h_best = h_opt.copy()
            
    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 10.0:
        h_cd, score_cd = _local_search_polish(h_best)
        if score_cd < score_best:
            score_best = score_cd
            h_best = h_cd.copy()
            
    # Phase 4: Second SLSQP pass with higher alpha for sharper max
    if time.time() < deadline - 5.0:
        h_opt2, score_opt2 = _optimize_slsqp(h_best, max_iter=300, alpha=100.0)
        if score_opt2 < score_best:
            score_best = score_opt2
            h_best = h_opt2.copy()
            
    # Final validity check
    h_best = _fix_sum(h_best)
    h_best = np.clip(h_best, 0.0, 1.0)
    
    return h_best
