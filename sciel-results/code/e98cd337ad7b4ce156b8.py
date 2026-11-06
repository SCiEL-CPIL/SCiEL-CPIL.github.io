import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 140.0  # Safety margin

# Precompute indices
_i_arr = np.arange(N, dtype=float)

def _value(h):
    """Compute the exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    # Fast autoconvolution via FFT
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = float(np.max(conv))
    val = abs(2.0 * N * m / (s * s))
    return val

def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate for the objective.
    Uses LogSumExp to approximate max(conv).
    """
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    # Shift to prevent overflow in exp
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _dct_h_odd(coeffs):
    """
    Reconstruct h from DCT-II coefficients where even indices > 0 are zero.
    coeffs: array of length K, where coeffs[k] corresponds to DCT mode k.
    We only use modes 0, 1, 3, 5, ...
    Mapping: coeffs[0] -> mode 0, coeffs[1] -> mode 1, coeffs[2] -> mode 3, etc.
    """
    full_coeffs = np.zeros(N)
    # coeffs[0] is mode 0
    full_coeffs[0] = coeffs[0]
    # coeffs[i] for i>=1 maps to mode 2*i - 1 (odd modes)
    num_odd = len(coeffs) - 1
    if num_odd > 0:
        odd_modes = np.arange(1, 2 * num_odd, 2)
        full_coeffs[odd_modes] = coeffs[1:]
        
    h = scipy_idct(full_coeffs, type=2, n=N, norm='ortho')
    return h

def _dct_odd_smooth_obj(coeffs, beta=50.0):
    """Objective for DCT odd-mode coefficients."""
    h = _dct_h_odd(coeffs)
    return _smooth_value(h, beta)

def _dct_odd_exact_obj(coeffs):
    """Exact objective for DCT odd-mode coefficients."""
    h = _dct_h_odd(coeffs)
    return _value(h)

def _generate_initial_dct_odd(rng, num_coeffs):
    """Generate random DCT odd-mode coefficients."""
    coeffs = np.zeros(num_coeffs)
    # Strong DC component
    coeffs[0] = rng.normal(0, 1.0) * 10.0
    
    # Random odd modes
    if num_coeffs > 1:
        # Decay weights to favor lower frequencies
        weights = 1.0 / (1.0 + np.arange(num_coeffs - 1))
        coeffs[1:] = rng.normal(0, 1.0, size=num_coeffs - 1) * weights * 2.0
        
    return coeffs

def construct():
    global _t0
    _t0 = time.time()
    
    # Start with the seed solution
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    rng = np.random.default_rng(42)
    
    # Strategy 1: Optimized DCT with Odd Modes (Antisymmetric structure)
    # This exploits the insight that antisymmetric components cancel the central peak
    # of the auto-convolution, allowing the max to be lower than the sum of squares.
    K = 400  # Use all modes, but constrain even ones to 0 in the reconstruction
    # Actually, we only pass the odd ones + DC.
    num_params = 1 + (N - 1) // 2  # DC + odd modes
    
    dct_restarts = 10
    
    for r in range(dct_restarts):
        if not deadline_ok():
            break
            
        try:
            coeffs0 = _generate_initial_dct_odd(rng, num_params)
            
            # Phase 1: Coarse optimization with smooth surrogate
            res = minimize(
                lambda c: _dct_odd_smooth_obj(c, beta=10.0),
                coeffs0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if not deadline_ok():
                break
                
            coeffs_opt = res.x if res.x is not None else coeffs0
            
            # Phase 2: Fine optimization with sharper surrogate
            if deadline_ok():
                res2 = minimize(
                    lambda c: _dct_odd_smooth_obj(c, beta=50.0),
                    coeffs_opt,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res2.x is not None:
                    coeffs_opt = res2.x

            h_candidate = _dct_h_odd(coeffs_opt)
            
            if np.all(np.isfinite(h_candidate)):
                update_best(h_candidate)
                
        except Exception:
            pass

    # Strategy 2: Full DCT optimization (Fallback / Exploration)
    # In case the odd-mode constraint is too restrictive
    K_full = 400
    full_restarts = 5
    
    def _dct_h_full(coeffs):
        return scipy_idct(coeffs, type=2, n=N, norm='ortho')

    def _dct_full_smooth_obj(coeffs, beta=50.0):
        h = _dct_h_full(coeffs)
        return _smooth_value(h, beta)

    for r in range(full_restarts):
        if not deadline_ok():
            break
            
        try:
            coeffs0 = np.zeros(K_full)
            coeffs0[0] = rng.normal(0, 1.0) * 10.0
            weights = 1.0 / (1.0 + np.arange(K_full))
            coeffs0[1:] = rng.normal(0, 1.0, size=K_full - 1) * weights[1:] * 2.0
            
            res = minimize(
                lambda c: _dct_full_smooth_obj(c, beta=20.0),
                coeffs0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if not deadline_ok():
                break
                
            coeffs_opt = res.x if res.x is not None else coeffs0
            
            if deadline_ok():
                res2 = minimize(
                    lambda c: _dct_full_smooth_obj(c, beta=100.0),
                    coeffs_opt,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res2.x is not None:
                    coeffs_opt = res2.x

            h_candidate = _dct_h_full(coeffs_opt)
            
            if np.all(np.isfinite(h_candidate)):
                update_best(h_candidate)
                
        except Exception:
            pass

    # Final check
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    # Simple self-test
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Shape: {h.shape}")
