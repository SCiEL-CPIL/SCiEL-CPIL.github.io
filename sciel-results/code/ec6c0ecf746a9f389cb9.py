"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_full(x):
    """Full linear autoconvolution of x with itself."""
    n = len(x)
    # Use FFT for efficiency
    nfft = 2 * n
    X = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(X * X, n=nfft)
    # The full autoconv is: ac[0:n] + reversed ac[n+1:2n]... 
    # Actually for full conv of length n with itself, result length is 2n-1
    # rfft gives us the right thing
    # ac[0] = sum x[i]^2
    # ac[k] for k=1..n-1 = sum x[i]*x[i+k]
    # ac[n+k] for k=0..n-1... hmm
    # Let me just use the standard approach
    # Full linear convolution: result[i] = sum_j x[j]*x[i-j] for valid j
    # Using FFT: ifft(fft(x)*fft(x)) gives circular conv, but with zero-padding it's linear
    # With nfft >= 2n-1, we get linear conv
    return ac[:2*n-1]

def _score(x):
    """Compute max |autoconv| / (sum x)^2"""
    ac = _autoconv_full(x)
    max_abs = np.max(np.abs(ac))
    s = np.sum(x)
    if abs(s) < 1e-15:
        return np.inf
    return max_abs / (s * s)

def _objective(x):
    """Objective for optimization: we want to minimize _score(x)
    But _score is scale-invariant, so we fix sum(x)=1 via penalty or reparametrization"""
    # Reparametrize: work with x, and the objective is scale-invariant
    # So we can just minimize max|ac| subject to sum(x)=1
    # Use a penalty for sum(x) != 1
    ac = _autoconv_full(x)
    max_abs = np.max(np.abs(ac))
    s = np.sum(x)
    penalty = 100.0 * (s - 1.0)**2
    return max_abs + penalty

def construct():
    deadline = time.time() + 100.0  # Safety margin
    best_x = np.ones(N) / N  # Normalized seed
    best_score = _score(best_x)
    
    # Try L-BFGS-B optimization
    from scipy.optimize import minimize
    
    x0 = np.ones(N) / N
    
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10})
        if time.time() > deadline - 10:
            return best_x
        if res.fun < _objective(x0):
            x_new = res.x
            s = np.sum(x_new)
            if abs(s) > 1e-10:
                x_new = x_new / s  # Normalize
            sc = _score(x_new)
            if sc < best_score:
                best_x = x_new
                best_score = sc
    except Exception:
        pass
    
    if time.time() > deadline - 5:
        return best_x
    
    # Try some heuristic improvements: random sign flips
    for _ in range(50):
        if time.time() > deadline - 5:
            break
        x_try = best_x.copy()
        # Random subset sign flip
        mask = np.random.random(N) < 0.3
        x_try[mask] = -x_try[mask]
        s = np.sum(x_try)
        if abs(s) < 1e-10:
            continue
        x_try = x_try / s
        sc = _score(x_try)
        if sc < best_score:
            best_x = x_try
            best_score = sc
    
    return best_x
# EVOLVE-BLOCK-END
