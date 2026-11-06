import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct as scipy_dct, idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # Safety margin

# Precompute indices
_i_arr = np.arange(N, dtype=float)

# Precompute atom templates for sparse representation - multiple diverse atoms
_ATOM_LEN = 10
_atom_templates = [
    np.hamming(_ATOM_LEN),
    np.hanning(_ATOM_LEN),
    np.ones(_ATOM_LEN) / _ATOM_LEN,  # boxcar
    np.cos(np.pi * np.arange(_ATOM_LEN) / _ATOM_LEN) ** 2,  # raised cosine
]
# Flatten to a single template list for indexing
_atom_template = np.hamming(_ATOM_LEN)  # Keep for backward compat in some functions

def _value(h):
    """Compute the exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    # Fast autoconvolution via FFT
    # Size needs to be at least 2*N - 1
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = float(np.max(conv))
    val = abs(2.0 * N * m / (s * s))
    return val

def _sparse_to_h(params):
    """
    Reconstruct h from sparse atom parameters.
    params: [amp_0, offset_0, sign_0, atom_idx_0, amp_1, offset_1, sign_1, atom_idx_1, ...]
    Each atom is placed at offset position with given amplitude, sign, and template index.
    """
    h = np.zeros(N)
    n_atoms = len(params) // 4
    for i in range(n_atoms):
        amp = params[4*i]
        offset = int(round(params[4*i + 1]))
        sign = params[4*i + 2]
        atom_idx = int(round(params[4*i + 3]))
        
        # Clamp offset to valid range
        offset = np.clip(offset, 0, N - _ATOM_LEN)
        # Clamp atom index to valid range
        atom_idx = np.clip(atom_idx, 0, len(_atom_templates) - 1)
        
        # Add atom to h
        h[offset:offset+_ATOM_LEN] += amp * sign * _atom_templates[atom_idx]
    
    return h

def _smooth_sparse_obj(params, beta=50.0):
    """Smooth surrogate for sparse atom objective."""
    h = _sparse_to_h(params)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    # LogSumExp approximation of max
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _exact_sparse_obj(params):
    """Exact objective for sparse atom parameters."""
    h = _sparse_to_h(params)
    return _value(h)

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
    # Clip to avoid numerical issues
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def _dct_h(coeffs):
    """Reconstruct h from DCT-II coefficients."""
    # scipy_idct with norm='ortho' is the inverse of scipy_dct with norm='ortho'
    h = scipy_idct(coeffs, type=2, n=N, norm='ortho')
    # Ensure zero mean? No, the sum is part of the variable. 
    # However, normalizing h to have sum N helps with scaling stability.
    # Let's allow the sum to vary but the optimizer will likely converge to a stable scale.
    return h

def _dct_smooth_obj(coeffs, beta=50.0):
    """Objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _smooth_value(h, beta)

def _dct_exact_obj(coeffs):
    """Exact objective for DCT coefficients."""
    h = _dct_h(coeffs)
    return _value(h)

def _generate_initial_dct(rng, num_coeffs):
    """Generate random DCT coefficients with low-frequency bias."""
    coeffs = np.zeros(num_coeffs)
    # Add a strong DC component (index 0) to ensure non-zero sum
    coeffs[0] = rng.normal(0, 1.0) * 10.0
    
    # Add random energy to other frequencies
    # Bias towards lower frequencies for smoother initial shapes, 
    # but allow high frequencies for sharp features
    weights = 1.0 / (1.0 + np.arange(num_coeffs))
    coeffs[1:] = rng.normal(0, 1.0, size=num_coeffs - 1) * weights[1:] * 2.0
    
    return coeffs

def _generate_initial_sparse(rng, n_atoms):
    """Generate random sparse atom parameters."""
    params = np.zeros(n_atoms * 4)
    for i in range(n_atoms):
        # Random amplitude
        params[4*i] = rng.normal(0, 1.0) * 2.0
        # Random offset
        params[4*i + 1] = rng.integers(0, N - _ATOM_LEN + 1)
        # Random sign
        params[4*i + 2] = rng.choice([-1.0, 1.0])
        # Random atom template index
        params[4*i + 3] = rng.integers(0, len(_atom_templates))
    return params

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
    
    # Strategy 1: DCT-based optimization (keep as fallback)
    K = 400
    dct_restarts = 10
    
    for r in range(dct_restarts):
        if not deadline_ok():
            break
            
        try:
            coeffs0 = _generate_initial_dct(rng, K)
            
            res = minimize(
                lambda c: _dct_smooth_obj(c, beta=20.0),
                coeffs0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if not deadline_ok():
                break
                
            coeffs_opt = res.x if res.x is not None else coeffs0
            
            if deadline_ok():
                res2 = minimize(
                    lambda c: _dct_smooth_obj(c, beta=100.0),
                    coeffs_opt,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res2.x is not None:
                    coeffs_opt = res2.x

            h_candidate = _dct_h(coeffs_opt)
            
            if np.all(np.isfinite(h_candidate)):
                update_best(h_candidate)
                
        except Exception:
            pass

    # Strategy 2: Sparse atom optimization (new)
    # Try different numbers of atoms
    for n_atoms in [5, 10, 15, 20]:
        if not deadline_ok():
            break
            
        sparse_restarts = 5
        
        for r in range(sparse_restarts):
            if not deadline_ok():
                break
                
            try:
                params0 = _generate_initial_sparse(rng, n_atoms)
                
                # Phase 1: Coarse optimization
                res = minimize(
                    lambda p: _smooth_sparse_obj(p, beta=20.0),
                    params0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if not deadline_ok():
                    break
                    
                params_opt = res.x if res.x is not None else params0
                
                # Phase 2: Fine optimization
                if deadline_ok():
                    res2 = minimize(
                        lambda p: _smooth_sparse_obj(p, beta=100.0),
                        params_opt,
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                    )
                    if res2.x is not None:
                        params_opt = res2.x

                # Phase 3: Very fine optimization with high beta
                if deadline_ok():
                    res3 = minimize(
                        lambda p: _smooth_sparse_obj(p, beta=300.0),
                        params_opt,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12}
                    )
                    if res3.x is not None:
                        params_opt = res3.x

                h_candidate = _sparse_to_h(params_opt)
                
                if np.all(np.isfinite(h_candidate)):
                    update_best(h_candidate)
                    
            except Exception:
                pass

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
