# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = 145.0  # wall-clock limit with safety margin
N = 400

def _evaluate_fast(h):
    """Compute the task value using FFT for speed."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    # Use FFT for convolution
    fft_h = np.fft.fft(h, n=2*N)
    conv = np.real(np.fft.ifft(fft_h * fft_h))
    # The full linear convolution is in the first 2N-1 elements
    # The last element (index 2N-1) is a wrap-around artifact
    max_conv = np.max(conv[:2*N-1])
    val = abs(2.0 * N * max_conv / (s * s))
    if not np.isfinite(val):
        return np.inf
    return val

def _build_from_sparse_spectrum(freq_indices, complex_coeffs):
    """
    Build h as IDFT of sparse spectrum.
    freq_indices: array of frequency bin indices (0..N-1)
    complex_coeffs: array of complex values for those bins
    """
    N_local = len(freq_indices)
    if N_local == 0:
        return np.zeros(N)
    
    # Create full spectrum with zeros
    spectrum = np.zeros(N, dtype=complex)
    for i, f in enumerate(freq_indices):
        f_mod = f % N
        spectrum[f_mod] += complex_coeffs[i]
    
    # IDFT to get time domain
    h = np.fft.ifft(spectrum).real
    return h

def _build_from_params(params):
    """
    Build h from optimized parameters.
    params layout: [scale, offset, n_freqs, f0, f1, ..., f_{n_freqs-1}, a0, b0, a1, b1, ...]
    where f_i are frequency indices, a_i + j*b_i are complex coefficients
    """
    scale = params[0]
    offset = params[1]
    n_freqs = int(round(params[2]))
    n_freqs = max(1, min(n_freqs, 20))
    
    if n_freqs > 20:
        n_freqs = 20
    
    # Extract frequency indices and complex coefficients
    freq_start = 3
    freqs = params[freq_start:freq_start+n_freqs].astype(int)
    freqs = np.clip(freqs, 0, N-1)
    
    coeff_start = freq_start + n_freqs
    coeffs = np.zeros(n_freqs, dtype=complex)
    for i in range(n_freqs):
        a = params[coeff_start + 2*i]
        b = params[coeff_start + 2*i + 1]
        coeffs[i] = a + 1j * b
    
    h = _build_from_sparse_spectrum(freqs, coeffs)
    h = scale * h + offset
    return h

def _objective(params):
    """
    Objective function for optimization.
    Returns the task value.
    """
    try:
        h = _build_from_params(params)
        if not np.all(np.isfinite(h)):
            return np.inf
        return _evaluate_fast(h)
    except Exception:
        return np.inf

def _local_refine(h, deadline, max_iter=50000):
    """
    Local refinement using direct perturbation.
    """
    best_val = _evaluate_fast(h)
    best_h = h.copy()
    rng = np.random.default_rng(42)
    
    step_size = 0.1
    for step in range(max_iter):
        if time.time() > deadline:
            break
        
        n_perturb = rng.integers(1, 6)
        idx = rng.choice(N, size=n_perturb, replace=False)
        h_new = best_h.copy()
        h_new[idx] += rng.normal(0, step_size, n_perturb)
        
        v_new = _evaluate_fast(h_new)
        if np.isfinite(v_new) and v_new < best_val - 1e-10:
            best_val = v_new
            best_h = h_new
            step_size *= 0.99
        elif step % 5000 == 0:
            step_size *= 0.95
    
    return best_h, best_val

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    rng = np.random.default_rng(123)
    
    best_val = 2.0
    best_h = np.ones(N)
    
    # Phase 1: Optimize sparse spectrum representation
    # Try different numbers of frequency bins
    n_freq_options = [2, 3, 4, 5, 6, 8, 10]
    
    for n_freqs in n_freq_options:
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(3):
            if time.time() > deadline - 25:
                break
            
            # Initialize frequency indices
            # Use low frequencies for smooth signals
            if trial == 0:
                # Very low frequencies
                freqs_init = rng.choice(N, size=n_freqs, replace=False)
                freqs_init = np.sort(freqs_init[:min(n_freqs, 10)])
            elif trial == 1:
                # Mix of low and mid frequencies
                freqs_init = np.concatenate([
                    rng.choice(20, size=min(n_freqs//2, 20), replace=False),
                    rng.choice(20, N-20, size=n_freqs - min(n_freqs//2, 20), replace=False) + 20
                ])
                freqs_init = np.sort(freqs_init[:n_freqs])
            else:
                # Random
                freqs_init = rng.choice(N, size=n_freqs, replace=False)
            
            # Initialize complex coefficients
            coeffs_init = np.zeros(n_freqs)
            for i in range(n_freqs):
                coeffs_init[i] = rng.uniform(0.5, 2.0) + 1j * rng.uniform(-1.0, 1.0)
            
            # Build initial params array
            # [scale, offset, n_freqs, f0, f1, ..., f_{n-1}, a0, b0, a1, b1, ...]
            params_init = np.zeros(3 + n_freqs + 2*n_freqs)
            params_init[0] = 1.0  # scale
            params_init[1] = 0.0  # offset
            params_init[2] = float(n_freqs)
            params_init[3:3+n_freqs] = freqs_init
            for i in range(n_freqs):
                params_init[3+n_freqs+2*i] = coeffs_init[i].real
                params_init[3+n_freqs+2*i+1] = coeffs_init[i].imag
            
            try:
                res = minimize(
                    _objective,
                    params_init,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                
                h_opt = _build_from_params(res.x)
                if not np.all(np.isfinite(h_opt)):
                    continue
                    
                v_opt = _evaluate_fast(h_opt)
                
                if np.isfinite(v_opt) and v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 2: Also try direct optimization with structured initialization
    # Use low-frequency cosine sums
    if time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            
            n = np.arange(N)
            h_struct = np.zeros(N)
            for k in range(3):
                freq = rng.uniform(0.5, 3.0)
                phase = rng.uniform(0, 2*np.pi)
                amp = rng.uniform(0.5, 2.0)
                h_struct += amp * np.cos(2 * np.pi * freq * n / N + phase)
            
            try:
                h_opt, v_opt = _local_refine(h_struct, deadline - 5, max_iter=8000)
                if np.isfinite(v_opt) and v_opt < best_val:
                    best_val = v_opt
                    best_h = h_opt.copy()
            except Exception:
                pass
    
    # Phase 3: Local refinement on best result
    if time.time() < deadline - 5:
        best_h, best_val = _local_refine(best_h, deadline - 2)
    
    # Final check
    final_val = _evaluate_fast(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
