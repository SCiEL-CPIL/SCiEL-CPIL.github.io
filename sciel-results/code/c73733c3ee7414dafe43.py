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
    dx = 0.01
    x = -0.25 + (np.arange(N) + 0.5) * dx
    
    # Generate diverse initial guesses
    np.random.seed(42)
    
    # Systematic grid of initial guesses
    init_guesses = []
    
    # Symmetric placements: pulses at ±d
    for d in [0.05, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]:
        for s in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10]:
            for b in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
                init_guesses.append([-d, d, s, s, b])
                init_guesses.append([d, -d, s, s, b])
    
    # Asymmetric placements
    for c1 in [-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20]:
        for c2 in [-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20]:
            if abs(c1 - c2) < 0.05:
                continue
            for s in [0.02, 0.04, 0.06, 0.08, 0.10]:
                for b in [0.5, 1.0, 2.0, 5.0]:
                    init_guesses.append([c1, c2, s, s, b])
    
    # Random additional starts
    for _ in range(200):
        c1 = np.random.uniform(-0.24, 0.24)
        c2 = np.random.uniform(-0.24, 0.24)
        if abs(c1 - c2) < 0.08:
            continue
        s1 = np.random.uniform(0.01, 0.12)
        s2 = np.random.uniform(0.01, 0.12)
        b = np.random.uniform(0.1, 10.0)
        init_guesses.append([c1, c2, s1, s2, b])
    
    best_params = None
    best_score = -np.inf
    
    # Evaluate all initial guesses and refine the best ones
    scored_guesses = []
    for guess in init_guesses:
        try:
            s = _compute_score(guess)
            if s > 0:
                scored_guesses.append((s, guess))
        except Exception:
            pass
    
    scored_guesses.sort(key=lambda x: -x[0])
    
    # Refine top candidates
    top_n = min(30, len(scored_guesses))
    for i in range(top_n):
        s_init, x0 = scored_guesses[i]
        try:
            res = minimize(_neg_score, x0, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < 1e10:
                s = -res.fun
                if s > best_score:
                    best_score = s
                    best_params = res.x
        except Exception:
            pass
    
    # If no good result, try more aggressive random search
    if best_params is None or best_score < 0.5:
        for _ in range(500):
            c1 = np.random.uniform(-0.24, 0.24)
            c2 = np.random.uniform(-0.24, 0.24)
            if abs(c1 - c2) < 0.06:
                continue
            s1 = np.random.uniform(0.005, 0.15)
            s2 = np.random.uniform(0.005, 0.15)
            b = np.random.uniform(0.05, 20.0)
            x0 = [c1, c2, s1, s2, b]
            try:
                res = minimize(_neg_score, x0, method='Nelder-Mead',
                              options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9})
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
    
    w1 = np.exp(-0.5 * ((x - c1) / s1) ** 2)
    w2 = np.exp(-0.5 * ((x - c2) / s2) ** 2)
    
    h = (w1 + b * w2) ** 2
    h = np.maximum(h, 0.0)
    
    return [float(v) for v in h]
