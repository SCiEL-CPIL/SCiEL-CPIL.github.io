"""Rewritten program for third autocorrelation inequality optimization."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.fft import dct, idct
from scipy.optimize import minimize
from scipy.special import eval_chebyu

def _eval_value(h):
    """Compute the objective value."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _dct_to_h(coeffs):
    """Convert DCT-II coefficients to time-domain signal."""
    h = idct(coeffs, type=2, norm='ortho')
    return h

def construct():
    """
    Returns a sequence of 400 signed step heights.
    Uses Chebyshev polynomial of the second kind parameterization.
    """
    deadline = time.time() + 140.0
    
    # Start with all-ones
    best_h = np.ones(N)
    best_val = 2.0
    
    # Chebyshev setup
    # Map i in [0, N-1] to x in [-1, 1]
    # Use midpoints of intervals
    x = 1.0 - 2.0 * (np.arange(N) + 0.5) / N
    
    M = 30  # Number of Chebyshev modes
    
    # Precompute U_k(x) for k=0..M-1
    # U_k is the Chebyshev polynomial of the second kind
    U = np.zeros((M, N))
    for k in range(M):
        U[k] = eval_chebyu(k, x)
        
    # Compute S_k = sum_i U_k(x_i)
    S = np.sum(U, axis=1)
    
    # We want sum(h) = 1.
    # h = sum_k c_k U_k
    # sum(h) = sum_k c_k S_k = 1
    # Let c_0 be the variable determined by the others.
    # c_0 S_0 + sum_{k=1}^{M-1} c_k S_k = 1
    # c_0 = (1 - sum_{k=1}^{M-1} c_k S_k) / S_0
    
    if abs(S[0]) < 1e-12:
        # Fallback to DCT if S_0 is zero (unlikely for this grid)
        return _construct_fallback(deadline)
        
    def _cheb_objective(c_ac):
        # c_ac are coefficients for k=1..M-1
        c = np.zeros(M)
        c[1:] = c_ac
        # Solve for c[0]
        c[0] = (1.0 - np.dot(c[1:], S[1:])) / S[0]
        
        # Reconstruct h
        h = np.dot(c, U)
        
        # Check validity
        if not np.all(np.isfinite(h)):
            return 1e10
            
        s = np.sum(h)
        if abs(s) < 1e-12:
            return 1e10
            
        # Calculate objective
        # Note: objective is scale invariant, but we fixed sum=1.
        # value = abs(2 * N * max(conv) / s^2)
        # Since s=1, value = abs(2 * N * max(conv))
        conv = np.convolve(h, h)
        max_conv = np.max(conv)
        return abs(2.0 * N * max_conv)
        
    # Initial guess for AC coefficients
    rng = np.random.RandomState(42)
    c_ac_init = 0.1 * rng.randn(M - 1)
    
    # Optimize
    try:
        res = minimize(_cheb_objective, c_ac_init, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        if res.success or res.fun < best_val:
            c_ac_opt = res.x
            c = np.zeros(M)
            c[1:] = c_ac_opt
            c[0] = (1.0 - np.dot(c[1:], S[1:])) / S[0]
            h_cheb = np.dot(c, U)
            v_cheb = _eval_value(h_cheb)
            if v_cheb < best_val:
                best_val = v_cheb
                best_h = h_cheb.copy()
    except Exception:
        pass
        
    # Try a few different random restarts if time permits
    if time.time() < deadline - 30:
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            c_ac_init = 0.5 * rng.randn(M - 1)
            try:
                res = minimize(_cheb_objective, c_ac_init, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                if res.fun < best_val:
                    c_ac_opt = res.x
                    c = np.zeros(M)
                    c[1:] = c_ac_opt
                    c[0] = (1.0 - np.dot(c[1:], S[1:])) / S[0]
                    h_cheb = np.dot(c, U)
                    v_cheb = _eval_value(h_cheb)
                    if v_cheb < best_val:
                        best_val = v_cheb
                        best_h = h_cheb.copy()
            except Exception:
                pass
                
    # Fallback to DCT coordinate descent if Chebyshev didn't improve much or time remains
    if time.time() < deadline - 20 and best_val > 1.1:
         # Simple DCT coordinate descent
         K = 20
         a = np.zeros(N)
         a[0] = 1.0 / np.sqrt(N)
         a[1:K] = 0.1 * rng.randn(K - 1)
         best_dct_val = _eval_value(_dct_to_h(a))
         if best_dct_val < best_val:
             best_val = best_dct_val
             best_h = _dct_to_h(a).copy()
             
         step = 0.05
         for iteration in range(50):
             if time.time() > deadline - 5:
                 break
             improved = False
             for k in range(1, K):
                 if time.time() > deadline - 5:
                     break
                 old = a[k]
                 for delta in [step, -step]:
                     a[k] = old + delta
                     h = _dct_to_h(a)
                     v = _eval_value(h)
                     if v < best_dct_val:
                         best_dct_val = v
                         if v < best_val:
                             best_val = v
                             best_h = h.copy()
                         improved = True
                     else:
                         a[k] = old
             if not improved:
                 step *= 0.5
                 if step < 0.001:
                     break

    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
        
    return best_h

def _construct_fallback(deadline):
    """Fallback using DCT if Chebyshev setup fails."""
    best_h = np.ones(N)
    best_val = 2.0
    rng = np.random.RandomState(42)
    
    K = 50
    a = np.zeros(N)
    a[0] = 1.0 / np.sqrt(N)
    a[1:K] = 0.1 * rng.randn(K - 1)
    
    best_dct_val = _eval_value(_dct_to_h(a))
    if best_dct_val < best_val:
        best_val = best_dct_val
        best_h = _dct_to_h(a).copy()
        
    step = 0.05
    for iteration in range(100):
        if time.time() > deadline - 5:
            break
        improved = False
        for k in range(1, K):
            if time.time() > deadline - 5:
                break
            old = a[k]
            for delta in [step, -step]:
                a[k] = old + delta
                h = _dct_to_h(a)
                v = _eval_value(h)
                if v < best_dct_val:
                    best_dct_val = v
                    if v < best_val:
                        best_val = v
                        best_h = h.copy()
                    improved = True
                else:
                    a[k] = old
        if not improved:
            step *= 0.5
            if step < 0.001:
                break
                
    return best_h
# EVOLVE-BLOCK-END
