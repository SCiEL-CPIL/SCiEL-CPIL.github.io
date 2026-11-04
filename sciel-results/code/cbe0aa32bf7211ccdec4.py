"""Optimized program for second autocorrelation inequality."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(h):
    """Compute the ratio: ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    The autoconvolution here is circular convolution of h with itself.
    """
    h = np.asarray(h, dtype=float)
    if np.any(h < -1e-12):
        return -np.inf
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return -np.inf
    l2_sq = np.sum(h**2)
    # Circular autoconvolution: h * h
    autoconv = np.fft.ifft(np.fft.fft(h)**2).real
    sup_norm = np.max(np.abs(autoconv))
    if sup_norm < 1e-15:
        return -np.inf
    return l2_sq / (l1 * sup_norm)

def _negative_score(h):
    """Negative of the score for minimization."""
    s = _score(h)
    if s == -np.inf:
        return 1e10
    return -s

def construct():
    """Construct 50 nonnegative heights to maximize the ratio."""
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    best_h = np.ones(N)
    best_score = _score(best_h)
    
    # Try multiple random initializations with L-BFGS-B
    rng = np.random.RandomState(42)
    
    # Strategy: Use sparse spectral mask approach
    # Represent h as |IFFT(mask * coeffs)|^2 where mask has K active bins
    # This enforces low-rank structure that can minimize autoconvolution peak
    
    K_values = [3, 5, 7, 10, 15, 20]
    
    for K in K_values:
        if time.time() > deadline:
            break
        
        for trial in range(5):
            if time.time() > deadline:
                break
            
            # Create random sparse spectral mask
            mask = np.zeros(N, dtype=complex)
            active_bins = rng.choice(N, size=K, replace=False)
            for idx in active_bins:
                # Random complex coefficient
                mag = rng.uniform(0.5, 2.0)
                phase = rng.uniform(0, 2*np.pi)
                mask[idx] = mag * np.exp(1j * phase)
            
            # Compute initial h from spectral mask
            h_init = np.abs(np.fft.ifft(mask))**2
            h_init = np.maximum(h_init, 0)
            # Normalize to avoid issues
            if np.sum(h_init) > 1e-10:
                h_init = h_init / np.sum(h_init) * N
            
            # Optimize using L-BFGS-B
            try:
                result = minimize(
                    _negative_score,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_score * -1:
                    h_opt = np.maximum(result.x, 0)
                    s = _score(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
            except Exception:
                pass
        
        # Also try direct optimization from random starts
        for trial in range(3):
            if time.time() > deadline:
                break
            
            # Random nonnegative vector
            h_init = rng.uniform(0, 1, size=N)
            if np.sum(h_init) < 1e-10:
                h_init = np.ones(N) / N
            
            try:
                result = minimize(
                    _negative_score,
                    h_init,
                    method='L-BFGS-B',
                    bounds=[(0, None)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < best_score * -1:
                    h_opt = np.maximum(result.x, 0)
                    s = _score(h_opt)
                    if s > best_score:
                        best_score = s
                        best_h = h_opt.copy()
            except Exception:
                pass
    
    # Final refinement: local search on best solution
    if time.time() < deadline - 10:
        h_refine = best_h.copy()
        for _ in range(200):
            if time.time() > deadline:
                break
            # Random perturbation
            idx = rng.randint(N)
            delta = rng.uniform(-0.05, 0.05) * max(0.01, h_refine[idx])
            h_new = h_refine.copy()
            h_new[idx] = max(0, h_refine[idx] + delta)
            
            s_new = _score(h_new)
            if s_new > best_score:
                best_score = s_new
                best_h = h_new.copy()
            else:
                h_refine = best_h.copy()
    
    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)
    
    return list(best_h)
