import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def _autoconv_max(f):
    """Compute max of linear autoconvolution."""
    conv = np.convolve(f, f, mode='full')
    return np.max(conv)

def _score(f):
    """Compute the ratio ||f||_2^2 / (||f||_1 * max(f*f))."""
    f = np.asarray(f, dtype=float)
    f = np.maximum(f, 0)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 0.0
    l2_sq = np.sum(f**2)
    amax = _autoconv_max(f)
    if amax < 1e-12:
        return 0.0
    return l2_sq / (l1 * amax)

def _neg_score_constrained(x):
    """Negative score for optimizer, with L1 normalization built in."""
    f = np.maximum(x, 0)
    s = np.sum(f)
    if s < 1e-12:
        return 0.0
    f = f / s
    return -_score(f)

def _optimize_full_space(x0, maxiter=400):
    """Optimize in full 50-dim space using L-BFGS-B."""
    x0 = np.maximum(x0, 0)
    if np.sum(x0) < 1e-12:
        x0 = np.ones(N)
    x0 = x0 / np.sum(x0)
    
    try:
        res = minimize(_neg_score_constrained, x0, method='L-BFGS-B',
                       bounds=[(0, None)] * N,
                       options={'maxiter': maxiter, 'ftol': 1e-9, 'gtol': 1e-7})
        f_opt = np.maximum(res.x, 0)
        if np.sum(f_opt) > 1e-12:
            f_opt = f_opt / np.sum(f_opt)
            return f_opt, _score(f_opt)
    except Exception:
        pass
    return None, 0.0

def _coordinate_descent(f, max_iter=200, n_trials=15):
    """Polish with coordinate descent using finer line search."""
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
            best_t = old_f
            best_local_score = best_score
            
            if old_f > 1e-15:
                t_factors = np.logspace(-3, 2, n_trials)
            else:
                t_factors = np.linspace(0, max(np.max(f)*3, 0.1), n_trials)
            
            for t_factor in t_factors:
                f_trial = f.copy()
                f_trial[i] = t_factor
                if np.sum(f_trial) < 1e-12:
                    continue
                f_trial = f_trial / np.sum(f_trial)
                s = _score(f_trial)
                if s > best_local_score + 1e-12:
                    best_local_score = s
                    best_t = f_trial[i]
                    improved = True
            f[i] = best_t
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
    start_time = time.time()
    best_f = np.ones(N) / N
    best_score = _score(best_f)
    
    np.random.seed(42)
    
    # Curated starting points
    starting_points = [
        np.ones(N) / N,
        np.array([1.0] + [0.0]*49),
        np.array([1.0, 1.0] + [0.0]*48),
    ]
    
    # Add random starts
    for _ in range(8):
        f_rand = np.random.exponential(1.0, N)
        f_rand = np.maximum(f_rand, 0)
        if np.sum(f_rand) > 1e-12:
            f_rand = f_rand / np.sum(f_rand)
            starting_points.append(f_rand)
    
    # Gaussian
    x = np.arange(N)
    f_gauss = np.exp(-((x - 24.5)**2) / (2*8**2))
    f_gauss = f_gauss / np.sum(f_gauss)
    starting_points.append(f_gauss)
    
    # Plateau
    f_plateau = np.zeros(N)
    f_plateau[18:32] = 1.0
    f_plateau = f_plateau / np.sum(f_plateau)
    starting_points.append(f_plateau)
    
    for x0 in starting_points:
        if time.time() - start_time > 90:
            break
        f_opt, s = _optimize_full_space(x0, maxiter=400)
        if f_opt is not None and s > best_score:
            best_f = f_opt
            best_score = s
    
    # Coordinate descent polish
    if time.time() - start_time < 110:
        best_f = _coordinate_descent(best_f, max_iter=150, n_trials=12)
        best_score = _score(best_f)
    
    # A couple more random starts
    np.random.seed(77)
    for _ in range(3):
        if time.time() - start_time > 112:
            break
        f_rand = np.random.exponential(1.0, N)
        f_rand = np.maximum(f_rand, 0)
        if np.sum(f_rand) < 1e-12:
            continue
        f_rand = f_rand / np.sum(f_rand)
        f_opt, s = _optimize_full_space(f_rand, maxiter=300)
        if f_opt is not None and s > best_score:
            best_f = f_opt
            best_score = s
    
    # Final polish
    if time.time() - start_time < 118:
        best_f = _coordinate_descent(best_f, max_iter=80, n_trials=10)
        best_score = _score(best_f)
    
    return list(best_f)
# EVOLVE-BLOCK-END
