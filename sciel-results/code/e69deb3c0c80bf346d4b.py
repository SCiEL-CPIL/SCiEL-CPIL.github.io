"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _spectral_objective(h, p=4):
    """
    Smooth spectral objective: approximate max_k |H[k]|^2 using power sum.
    H[0] = sum(h), so we work with normalized h (sum=1).
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    h_norm = h / s
    H = np.fft.rfft(h_norm, N)
    # H[0] = 1, look at H[1:]
    H_mag_sq = np.abs(H[1:])**2
    if len(H_mag_sq) == 0:
        return 0.0
    # Power sum approximation to max
    return np.sum(H_mag_sq**p) / len(H_mag_sq)

def _spectral_optimize_direct():
    """
    Direct spectral optimization using smooth objective in frequency domain.
    Minimizes approximate max |H[k]|^2 subject to sum(h) = 1.
    """
    t_start = time.time()
    deadline = t_start + 100.0
    
    i = np.arange(N)
    
    # Multiple initializations
    initializations = []
    
    # Chirp-like
    for alpha in [0.5, 1.0, 1.5, 2.0]:
        h = np.cos(2.0 * np.pi * alpha * i**2 / N)
        if abs(np.sum(h)) > 1e-6:
            h = h / np.sum(h)
            initializations.append(h)
    
    # Linear phase
    for alpha in [0.5, 1.0]:
        h = np.cos(np.pi * alpha * i * (i + 1) / N)
        if abs(np.sum(h)) > 1e-6:
            h = h / np.sum(h)
            initializations.append(h)
    
    # Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        h = rng.randn(N)
        h = h - np.mean(h)
        h = h + 1.0 / N
        initializations.append(h)
    
    best_h = None
    best_score = np.inf
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}]
    
    for h_init in initializations:
        if time.time() > deadline - 20:
            break
        
        try:
            result = minimize(
                _spectral_objective,
                h_init,
                method='SLSQP',
                constraints=constraints,
                options={'maxiter': 30, 'ftol': 1e-8}
            )
            
            if result.success:
                h_opt = result.x
                h_opt = h_opt / np.sum(h_opt)
                score = _score(h_opt)
                
                if score < best_score:
                    best_score = score
                    best_h = h_opt.copy()
        except Exception:
            continue
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_score = _score(best_h)
    
    # Local search refinement
    if time.time() < deadline - 10:
        h_refined = best_h.copy()
        best_local = best_score
        for _ in range(5):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 5:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < best_score:
            best_score = best_local
            best_h = h_refined.copy()
    
    return best_h, best_score

def construct():
    t_start = time.time()
    deadline = t_start + 110.0
    
    h_spec, score_spec = _spectral_optimize_direct()
    
    # Final local search if time permits
    if time.time() < deadline - 5:
        h_refined = h_spec.copy()
        best_local = score_spec
        for _ in range(3):
            improved = False
            for idx in range(N):
                if time.time() > deadline - 3:
                    break
                h_refined[idx] = -h_refined[idx]
                new_val = _score(h_refined)
                if new_val < best_local:
                    best_local = new_val
                    improved = True
                else:
                    h_refined[idx] = -h_refined[idx]
            if not improved:
                break
        
        if best_local < score_spec:
            h_spec = h_refined.copy()
            score_spec = best_local
    
    return h_spec

# EVOLVE-BLOCK-END
