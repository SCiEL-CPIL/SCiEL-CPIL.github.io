"""
Strategy: Low-Frequency DFT Parameterization with Nelder-Mead Optimization.

Instead of optimizing 400 time-domain values directly, we parameterize h
as the inverse DFT of a sparse spectrum with K active frequency bins.
This reduces the optimization to ~2K real parameters (amplitude and phase
for each bin), making it feasible to use robust derivative-free methods.

The autoconvolution of h corresponds to |H|^2 in the frequency domain
(upsampled), so by choosing which frequency bins to activate and their
amplitudes/phases, we can shape the autoconvolution to minimize the
maximum peak relative to the sum squared.

We use:
- DFT parameterization with K modes (K ~ 20-40)
- Nelder-Mead for low-dimensional optimization
- Log-sum-exp surrogate for smooth max approximation
- Multiple restarts with different mode sets and initializations
- Final coordinate-descent polish in time domain
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 400
_DEADLINE = 135.0  # Safety margin before 150s limit


def _evaluate(h):
    """Compute the exact task value."""
    s = np.sum(h)
    if s == 0 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))


def _surrogate_max(x, alpha):
    """Smooth approximation of max(x)."""
    x_max = np.max(x)
    if not np.isfinite(x_max):
        return x_max
    x_shifted = (x - x_max) / alpha
    x_shifted = np.clip(x_shifted, -200, 200)
    log_sum_exp = np.log(np.sum(np.exp(x_shifted)))
    return x_max + alpha * log_sum_exp


def _h_from_dft_coeffs(coeffs, K):
    """
    Reconstruct h from DFT coefficients.
    coeffs: complex array of length K, representing bins 0..K-1.
    We enforce Hermitian symmetry for real output.
    """
    # Build full spectrum of length N
    H = np.zeros(N, dtype=complex)
    for k in range(K):
        H[k] = coeffs[k]
        # Hermitian conjugate: H[N-k] = conj(H[k])
        if k != 0 and k != N - k:
            H[N - k] = np.conj(coeffs[k])
    
    # For DC component (k=0), H[0] should be real
    H[0] = complex(coeffs[0].real, 0.0)
    # For Nyquist (if N even and k=N/2), H[N/2] should be real
    if N % 2 == 0 and K > N // 2:
        H[N // 2] = complex(coeffs[N // 2].real, 0.0)
    
    h = np.fft.ifft(H).real
    return h


def _objective_dft(params, K, alpha):
    """
    Objective function for DFT parameterization.
    params: real array of length 2*K (amplitude and phase for each bin).
    """
    # Convert params to complex coefficients
    amps = params[:K]
    phases = params[K:]
    coeffs = amps * np.exp(1j * phases)
    
    # Reconstruct h
    h = _h_from_dft_coeffs(coeffs, K)
    
    # Check validity
    s_h = np.sum(h)
    if abs(s_h) < 1e-8:
        return 1e10
    
    # Normalize for scale invariance
    h_norm = h / s_h
    
    # Compute autoconvolution
    conv_h = np.convolve(h_norm, h_norm)
    
    # Surrogate max
    peak = _surrogate_max(conv_h, alpha)
    
    # Small regularization to avoid degenerate solutions
    reg = 0.001 * np.sum(h_norm ** 2)
    
    return peak + reg


def construct():
    """
    Construct step heights using low-frequency DFT parameterization.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # Start with the seed result
    best_h = np.ones(N).copy()
    best_val = _evaluate(best_h)
    
    rng = np.random.default_rng(123)
    
    # Try different numbers of DFT modes
    K_values = [10, 15, 20, 25, 30, 35, 40]
    n_restarts_per_K = 4
    alpha_start = 1.0
    alpha_end = 0.1
    
    total_restarts = len(K_values) * n_restarts_per_K
    restart_count = 0
    
    for K in K_values:
        if time.time() > deadline - 30:
            break
        
        for r in range(n_restarts_per_K):
            if time.time() > deadline - 15:
                break
            
            restart_count += 1
            alpha = alpha_start + (alpha_end - alpha_start) * (restart_count / max(total_restarts, 1))
            
            # Initialize parameters
            # Option: random amplitudes and phases, or structured
            if r == 0:
                # Start with only DC + a few low modes
                amps_init = np.zeros(K)
                amps_init[0] = 1.0  # DC component
                for k in range(1, min(5, K)):
                    amps_init[k] = 0.5 * rng.random()
                phases_init = rng.uniform(0, 2 * np.pi, K)
            elif r == 1:
                # Random amplitudes
                amps_init = rng.uniform(0.1, 1.0, K)
                phases_init = rng.uniform(0, 2 * np.pi, K)
            elif r == 2:
                # Decay amplitudes
                k_arr = np.arange(K)
                amps_init = np.exp(-k_arr / (K / 3.0)) * rng.uniform(0.5, 1.5, K)
                phases_init = rng.uniform(0, 2 * np.pi, K)
            else:
                # Sparse: only a few active bins
                amps_init = np.zeros(K)
                n_active = max(3, K // 4)
                active_idx = rng.choice(K, size=n_active, replace=False)
                amps_init[active_idx] = rng.uniform(0.5, 1.0, n_active)
                phases_init = rng.uniform(0, 2 * np.pi, K)
            
            params_init = np.concatenate([amps_init, phases_init])
            
            try:
                # Use Nelder-Mead for derivative-free optimization
                res = minimize(
                    _objective_dft,
                    params_init,
                    args=(K, alpha),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 300,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if res.fun < 1e9:
                    amps_opt = res.x[:K]
                    phases_opt = res.x[K:]
                    coeffs_opt = amps_opt * np.exp(1j * phases_opt)
                    h_opt = _h_from_dft_coeffs(coeffs_opt, K)
                    
                    s_sum = np.sum(h_opt)
                    if abs(s_sum) > 1e-8:
                        h_norm = h_opt / s_sum
                        v_opt = _evaluate(h_norm)
                        if v_opt < best_val:
                            best_val = v_opt
                            best_h = h_norm.copy()
                            
            except Exception:
                pass
    
    # Additional: Try direct coordinate descent on the best solution found
    if time.time() < deadline - 5:
        h_curr = best_h.copy()
        v_curr = best_val
        step_size = 0.1
        
        for _ in range(300):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            h_new = h_curr.copy()
            h_new[idx] += rng.normal(0, step_size)
            v_new = _evaluate(h_new)
            if v_new < v_curr:
                v_curr = v_new
                h_curr = h_new
                step_size *= 0.97
            else:
                step_size = min(step_size * 1.05, 0.3)
        
        if v_curr < best_val:
            best_val = v_curr
            best_h = h_curr
    
    # Validate final result
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
