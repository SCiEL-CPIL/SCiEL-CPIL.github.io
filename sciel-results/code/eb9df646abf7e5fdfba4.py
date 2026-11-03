"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import time

def _overlap_max(h):
    """Compute max overlap of h with shifted 1-h."""
    n = len(h)
    max_ov = 0.0
    for k in range(n):
        shifted = np.roll(h, k)
        ov = np.sum(h * (1.0 - shifted))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _soft_max_overlaps(h, tau=0.5):
    """Smooth surrogate for max overlap using log-sum-exp."""
    n = len(h)
    overlaps = np.zeros(n)
    for k in range(n):
        shifted = np.roll(h, k)
        overlaps[k] = np.sum(h * (1.0 - shifted))
    # log-sum-exp for max
    m = np.max(overlaps)
    if tau < 1e-10:
        return m
    return m + tau * np.log(np.sum(np.exp((overlaps - m) / tau)))

def construct():
    deadline = time.time() + 110  # safety margin
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    
    # Spectral parameterization: h_i = 0.5 + sum of DFT modes
    # Use modes 1..M to keep search space small
    M = 8  # number of frequency modes
    
    def coeffs_to_h(coeffs):
        """Convert spectral coefficients to height sequence."""
        # coeffs: [a1, b1, a2, b2, ..., aM, bM]
        i = np.arange(N)
        h = np.full(N, 0.5)
        for m in range(1, M + 1):
            a = coeffs[2*(m-1)]
            b = coeffs[2*(m-1)+1]
            h += a * np.cos(2*np.pi*m*i/N) + b * np.sin(2*np.pi*m*i/N)
        return h
    
    def objective(coeffs):
        h = coeffs_to_h(coeffs)
        # Penalize out-of-bounds
        penalty = np.sum(np.maximum(0, h - 1.0)**2) + np.sum(np.maximum(0, -h)**2)
        if time.time() > deadline:
            return 1e10
        return _soft_max_overlaps(h, tau=0.3) + 100.0 * penalty
    
    # Multi-start optimization
    rng = np.random.RandomState(42)
    n_starts = 20
    
    for s in range(n_starts):
        if time.time() > deadline:
            break
        
        # Initialize with random small coefficients
        scale = 0.1 * (1.0 + s / n_starts)
        x0 = rng.randn(2*M) * scale
        
        # Use Nelder-Mead for robustness, then polish with L-BFGS-B
        try:
            res = opt.minimize(objective, x0, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            x_best = res.x
        except Exception:
            continue
        
        # Polish with L-BFGS-B
        try:
            res2 = opt.minimize(objective, x_best, method='L-BFGS-B',
                                options={'maxiter': 200, 'ftol': 1e-10})
            x_best = res2.x
        except Exception:
            pass
        
        h_candidate = coeffs_to_h(x_best)
        # Clip to [0,1] and renormalize sum to N/2
        h_candidate = np.clip(h_candidate, 0.0, 1.0)
        # Adjust to make sum exactly N/2
        diff = N/2.0 - np.sum(h_candidate)
        # Distribute adjustment uniformly if possible
        if diff != 0:
            # Try to adjust without going out of bounds
            adj = diff / N
            h_candidate = h_candidate + adj
            h_candidate = np.clip(h_candidate, 0.0, 1.0)
            # Iterate to fix sum
            for _ in range(10):
                s_sum = np.sum(h_candidate)
                if abs(s_sum - N/2.0) < 1e-10:
                    break
                adj = (N/2.0 - s_sum) / N
                h_candidate = h_candidate + adj
                h_candidate = np.clip(h_candidate, 0.0, 1.0)
        
        score = _overlap_max(h_candidate)
        if score < best_score:
            best_score = score
            best_h = h_candidate.copy()
    
    # Final polish: local search on the best solution
    if time.time() < deadline - 10:
        # Try small perturbations
        for trial in range(50):
            if time.time() > deadline - 5:
                break
            # Random small perturbation in spectral space
            perturb = rng.randn(2*M) * 0.01
            h_test = coeffs_to_h(np.array([0.0]*2*M) + perturb)  # This won't work well
            
            # Instead, do direct local search on h
            h_test = best_h + rng.randn(N) * 0.01
            h_test = np.clip(h_test, 0.0, 1.0)
            # Fix sum
            diff = N/2.0 - np.sum(h_test)
            h_test += diff / N
            h_test = np.clip(h_test, 0.0, 1.0)
            # Re-fix sum
            for _ in range(5):
                diff = N/2.0 - np.sum(h_test)
                if abs(diff) < 1e-10:
                    break
                h_test += diff / N
                h_test = np.clip(h_test, 0.0, 1.0)
            
            s_test = _overlap_max(h_test)
            if s_test < best_score:
                best_score = s_test
                best_h = h_test.copy()
    
    # Ensure final result has exact sum N/2 and is in [0,1]
    best_h = np.clip(best_h, 0.0, 1.0)
    for _ in range(20):
        diff = N/2.0 - np.sum(best_h)
        if abs(diff) < 1e-12:
            break
        best_h += diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
