"""Minimize normalized max autoconvolution via cosine-basis Nelder-Mead search.

Structural change: instead of 600-dim projected gradient descent, we parameterize
h as a sum of the first K cosine modes (plus a DC offset), ensuring smoothness and
band-limiting. We then use scipy.optimize.minimize with Nelder-Mead (derivative-free)
over the K+1 coefficients, with multi-start from different random initializations.
This explores a fundamentally different search landscape: smooth low-frequency shapes
where the autoconvolution peak can be made more uniform.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 600
DELTA = 0.5 / N
DEADLINE = 115.0  # safety margin before 150s kill

# Cosine basis: phi_k(x) = cos(k * pi * x / (0.5)) for k = 0, 1, ..., K-1
# On [-1/4, 1/4], let t = (x + 1/4) / (1/2) in [0, 1]
# phi_k(t) = cos(k * pi * t)
K_MAX = 20  # number of cosine coefficients to optimize

def _build_basis(K):
    """Precompute cosine basis vectors for K modes."""
    t = np.linspace(0.0, 1.0, N)
    basis = np.zeros((K, N))
    for k in range(K):
        basis[k] = np.cos(k * np.pi * t)
    return basis

def _objective_from_coeffs(coeffs, basis):
    """Compute F(h) = max(c) / (sum h)^2 from cosine coefficients."""
    h = basis @ coeffs
    h = np.clip(h, 0.0, None)  # enforce nonnegativity
    s = float(np.sum(h))
    if s <= 0:
        return 1e10
    c = np.convolve(h, h, mode='full')
    return float(np.max(c)) / (s * s)

def _objective_vec(coeffs, basis):
    """Vectorized objective for scipy.optimize."""
    return _objective_from_coeffs(coeffs, basis)

def construct():
    global _t0
    _t0 = time.time()
    
    # Precompute basis
    basis = _build_basis(K_MAX)
    
    # Multi-start: try several different initial coefficient vectors
    best_val = float('inf')
    best_coeffs = None
    
    # Start 0: all zeros except DC (uniform)
    # DC coefficient: h = c0 * ones, so c0 = 1 gives h = ones
    # But our basis[0] = cos(0) = 1, so coeffs = [1, 0, 0, ...] gives h = ones
    init_candidates = []
    
    # Uniform: [1, 0, 0, ..., 0]
    init_candidates.append(np.zeros(K_MAX))
    init_candidates[0][0] = 1.0
    
    # Raised cosine: h(t) = 0.5*(1 + cos(pi*t)) = 0.5*basis[0] + 0.5*basis[1]
    rc = np.zeros(K_MAX)
    rc[0] = 0.5
    rc[1] = 0.5
    init_candidates.append(rc)
    
    # Tent-like: h(t) = 1 - |2t - 1| ≈ DC + some harmonics
    # Approximate with first few cosines
    tent = np.zeros(K_MAX)
    tent[0] = 0.5
    tent[1] = 0.1
    tent[2] = -0.05
    tent[3] = 0.02
    init_candidates.append(tent)
    
    # Random smooth starts
    rng = np.random.default_rng(42)
    for trial in range(8):
        if time.time() - _t0 > DEADLINE - 30:
            break
        # Random coefficients with decay for smoothness
        c = rng.normal(0, 1, K_MAX) * np.exp(-np.arange(K_MAX) * 0.3)
        c[0] = max(c[0], 0.5)  # ensure positive DC
        init_candidates.append(c)
    
    # Evaluate all initial candidates
    for c in init_candidates:
        if time.time() - _t0 > DEADLINE - 20:
            break
        v = _objective_from_coeffs(c, basis)
        if v < best_val:
            best_val = v
            best_coeffs = c.copy()
    
    # Nelder-Mead optimization from each promising start
    starts_to_optimize = []
    for c in init_candidates:
        v = _objective_from_coeffs(c, basis)
        if v < best_val * 1.5:  # only optimize from reasonably good starts
            starts_to_optimize.append(c)
    
    # Limit number of Nelder-Mead runs to stay within time budget
    max_nm_runs = 6
    nm_runs_done = 0
    
    for c0 in starts_to_optimize[:max_nm_runs]:
        if time.time() - _t0 > DEADLINE - 10:
            break
        if nm_runs_done >= max_nm_runs:
            break
        
        # Nelder-Mead with adaptive settings
        try:
            res = minimize(
                _objective_vec,
                c0.copy(),
                args=(basis,),
                method='Nelder-Mead',
                options={
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'maxiter': 500,
                    'maxfev': 2000,
                    'adaptive': True,
                }
            )
            nm_runs_done += 1
            
            v = _objective_from_coeffs(res.x, basis)
            if v < best_val:
                best_val = v
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Also try a few more random starts with shorter NM runs if time permits
    if time.time() - _t0 < DEADLINE - 15:
        for trial in range(5):
            if time.time() - _t0 > DEADLINE - 5:
                break
            c = rng.normal(0, 1, K_MAX) * np.exp(-np.arange(K_MAX) * 0.2)
            c[0] = max(c[0], 0.3)
            
            try:
                res = minimize(
                    _objective_vec,
                    c.copy(),
                    args=(basis,),
                    method='Nelder-Mead',
                    options={
                        'xatol': 1e-5,
                        'fatol': 1e-7,
                        'maxiter': 300,
                        'maxfev': 1000,
                        'adaptive': True,
                    }
                )
                v = _objective_from_coeffs(res.x, basis)
                if v < best_val:
                    best_val = v
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Convert best coefficients to heights
    if best_coeffs is not None:
        h = basis @ best_coeffs
        h = np.clip(h, 0.0, None)
    else:
        h = np.ones(N)
    
    # Safety: ensure we don't return something worse than uniform
    uniform_val = _objective_from_coeffs(np.zeros(K_MAX), basis)
    # Actually uniform is just [1,0,...,0]
    uniform_val = float(np.max(np.convolve(np.ones(N), np.ones(N), mode='full'))) / (N * N)
    
    if best_val > uniform_val + 1e-12:
        h = np.ones(N)
    
    return [float(v) for v in h]


_t0 = time.time()
