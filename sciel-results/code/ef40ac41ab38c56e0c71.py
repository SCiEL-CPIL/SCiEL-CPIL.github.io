"""
Design: Represent 50 heights as squared modulus of a sum of two orthogonal,
disjointly supported windowed basis functions.

h_k = |a * w1(k) + b * w2(k)|^2

where w1, w2 are non-overlapping Gaussian-like pulses separated by a large gap.
Since they are disjointly supported, |a*w1 + b*w2|^2 = a^2 * w1^2 + b^2 * w2^2
(no cross terms). This makes the autoconvolution additive in a useful way.

We optimize: separation distance, widths (sigma1, sigma2), and amplitudes (a, b)
to maximize the ratio: ||h||_2^2 / (||h||_1 * ||h * h||_inf)

Note: The heights are on equal intervals of [-1/4, 1/4], so 50 points.
The step size is (1/2)/50 = 1/100 = 0.01.

We use a global search (random restarts + local refinement) over the parameters.
"""

import numpy as np
from scipy.optimize import minimize

N = 50

def _compute_score(params):
    """
    params: [c1, c2, s1, s2, b]
    a is fixed to 1.0.
    Returns the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    """
    c1, c2, s1, s2, b = params
    
    s1 = max(abs(s1), 0.005)
    s2 = max(abs(s2), 0.005)
    b = abs(b)
    
    if b < 1e-10:
        return -1.0
        
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    if abs(c1 - c2) < (s1 + s2) * 3.0:
        return -1.0

    dx = 0.01
    x = -0.25 + (np.arange(N) + 0.5) * dx
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    h = (w1 + b * w2) ** 2
    h = np.maximum(h, 0.0)
    
    l1 = np.sum(h) * dx
    if l1 <= 0:
        return -1.0
        
    l2_sq = np.sum(h**2) * dx
    
    conv = np.convolve(h, h, mode='full') * dx
    sup_norm = np.max(conv)
    
    if sup_norm <= 0:
        return -1.0
        
    return l2_sq / (l1 * sup_norm)

def _neg_score(params):
    s = _compute_score(params)
    if s < 0:
        return 1e10
    return -s

def construct():
    bounds = [
        (-0.25, 0.25),
        (-0.25, 0.25),
        (0.005, 0.15),
        (0.005, 0.15),
        (0.1, 5.0)
    ]
    
    best_params = None
    best_score = -np.inf
    
    np.random.seed(42)
    
    num_starts = 100
    
    for i in range(num_starts):
        x0 = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        
        if i % 2 == 0:
            x0[0] = -0.15
            x0[1] = 0.15
        else:
            x0[0] = 0.15
            x0[1] = -0.15
            
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < 1e10:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        return [1.0] * N
        
    c1, c2, s1, s2, b = best_params
    s1 = max(abs(s1), 0.005)
    s2 = max(abs(s2), 0.005)
    b = abs(b)
    c1 = np.clip(c1, -0.25, 0.25)
    c2 = np.clip(c2, -0.25, 0.25)
    
    dx = 0.01
    x = -0.25 + (np.arange(N) + 0.5) * dx
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    h = (w1 + b * w2) ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
