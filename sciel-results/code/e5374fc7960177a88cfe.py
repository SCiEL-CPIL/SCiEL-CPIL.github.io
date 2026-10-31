import numpy as np
from scipy.optimize import minimize
from itertools import product

N = 50

# EVOLVE-BLOCK-START
def _autoconv_max(f):
    """Compute max of circular autoconvolution (or linear, since N=50 is small)."""
    # Linear autoconvolution
    conv = np.convolve(f, f, mode='full')
    return np.max(conv)

def _score(f):
    """Compute the ratio ||f||_2^2 / (||f||_1 * max(f*f))."""
    f = np.asarray(f, dtype=float)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(f**2)
    amax = _autoconv_max(f)
    if amax < 1e-12:
        return 0.0
    return l2_sq / (l1 * amax)

def _triangular_symmetric(params):
    """Symmetric triangle centered at 24.5 with half-width L."""
    L = params[0]
    f = np.zeros(N)
    center = 24.5
    for i in range(N):
        dist = abs(i - center)
        if dist < L:
            f[i] = 1.0 - dist / L
    return f

def _asymmetric_ramp(params):
    """Ramp from 0 to 1 over [a, a+L]."""
    a = params[0]
    L = params[1]
    f = np.zeros(N)
    for i in range(N):
        if a <= i <= a + L:
            f[i] = (i - a) / L
    return f

def _gaussian(params):
    """Gaussian centered at mu with std sigma."""
    mu = params[0]
    sigma = params[1]
    f = np.zeros(N)
    for i in range(N):
        f[i] = np.exp(-((i - mu)**2) / (2 * sigma**2))
    return f

def _optimize_family(family_func, bounds, x0):
    """Optimize a parametric family using Nelder-Mead."""
    def neg_score(params):
        f = family_func(params)
        if np.sum(f) < 1e-12:
            return 0.0
        # Normalize to L1=1 for consistency
        f_norm = f / np.sum(f)
        return -_score(f_norm)
    
    try:
        res = minimize(neg_score, x0, method='Nelder-Mead',
                       bounds=bounds,
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8})
        f_opt = family_func(res.x)
        if np.sum(f_opt) > 1e-12:
            f_opt = f_opt / np.sum(f_opt)
            s = _score(f_opt)
            return f_opt, s
    except Exception:
        pass
    return None, 0.0

def _coordinate_descent(f, max_iter=200):
    """Polish with coordinate descent, keeping nonnegativity and L1 normalization."""
    f = np.array(f, dtype=float)
    f = np.maximum(f, 0)
    if np.sum(f) < 1e-12:
        return f
    f = f / np.sum(f)
    best_score = _score(f)
    
    for _ in range(max_iter):
        improved = False
        for i in range(N):
            old_f = f[i].copy()
            # Try varying f[i] while keeping others fixed
            # We can scale f[i] by factor t >= 0
            # To keep L1 = 1, we need to renormalize, but that changes all entries.
            # Instead, do a line search: for t in a grid, set f[i] = t * old_f, 
            # then renormalize the whole vector to L1=1.
            best_t = old_f
            best_local_score = best_score
            for t_factor in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0, 3.0]:
                f_trial = f.copy()
                f_trial[i] = old_f * t_factor
                if np.sum(f_trial) < 1e-12:
                    continue
                f_trial = f_trial / np.sum(f_trial)
                s = _score(f_trial)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_t = f_trial[i]
                    improved = True
            f[i] = best_t
            # Renormalize after each coordinate update
            if np.sum(f) > 1e-12:
                f = f / np.sum(f)
            else:
                break
            if best_local_score > best_score + 1e-12:
                best_score = best_local_score
        if not improved:
            break
    return f

def construct():
    best_f = np.ones(N)
    best_score = _score(best_f / np.sum(best_f))
    
    families = []
    
    # Symmetric triangular: L in [3, 25]
    families.append((_triangular_symmetric, [(3.0, 25.0)], np.array([12.0])))
    
    # Asymmetric ramp: a in [0, 30], L in [3, 25]
    families.append((_asymmetric_ramp, [(0.0, 30.0), (3.0, 25.0)], np.array([10.0, 15.0])))
    
    # Gaussian: mu in [5, 45], sigma in [2, 15]
    families.append((_gaussian, [(5.0, 45.0), (2.0, 15.0)], np.array([24.5, 5.0])))
    
    for family_func, bounds, x0 in families:
        f_opt, s = _optimize_family(family_func, bounds, x0)
        if f_opt is not None and s > best_score:
            best_f = f_opt
            best_score = s
    
    # Coordinate descent polish
    best_f = _coordinate_descent(best_f, max_iter=100)
    best_score = _score(best_f)
    
    # Also try starting from uniform and polishing
    f_uniform = np.ones(N) / N
    f_polished = _coordinate_descent(f_uniform, max_iter=100)
    s_uniform = _score(f_polished)
    if s_uniform > best_score:
        best_f = f_polished
        best_score = s_uniform
    
    # Try a few random initializations
    np.random.seed(42)
    for _ in range(20):
        f_rand = np.random.exponential(1.0, N)
        f_rand = np.maximum(f_rand, 0)
        if np.sum(f_rand) < 1e-12:
            continue
        f_rand = f_rand / np.sum(f_rand)
        f_polished = _coordinate_descent(f_rand, max_iter=50)
        s = _score(f_polished)
        if s > best_score:
            best_f = f_polished
            best_score = s
    
    return list(best_f)
# EVOLVE-BLOCK-END
