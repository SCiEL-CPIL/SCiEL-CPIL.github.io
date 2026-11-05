"""
Strategy: FIR Low-Pass Filter Parameterization with Adaptive Smooth-Max

The step heights h are represented as the impulse response of a low-pass FIR filter
with coefficients in the frequency domain. We exploit the fact that the autoconvolution
c = h * h has a Fourier transform C = |H|^2, and the peak of c relates to the peak
of the inverse Fourier transform of |H|^2.

We parameterize h via a small number of Fourier coefficients (DCT basis) to reduce
the effective dimensionality, then optimize both the Fourier coefficients and an
adaptive smooth-max temperature using L-BFGS-B.

The smooth-max surrogate: soft_max(x, tau) = tau * log(sum(exp(x_i/tau)))
approximates max(x) when tau is small. We use a decreasing schedule of tau during
optimization to progressively sharpen the max.

Key insight: Low-frequency (smooth) step functions have autoconvolutions that are
also smooth, and the peak is typically at lag 0 or near it. By restricting to
low-frequency content via the DCT parameterization, we naturally suppress high-frequency
oscillations that would create sharp peaks in the autoconvolution.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.fft import dct, idct

_DEADLINE = 140.0  # wall-clock limit with safety margin

def _evaluate(h):
    """Compute the task value: abs(2 * N * max(convolve(h,h)) / sum(h)^2)."""
    s = np.sum(h)
    if abs(s) < 1e-12 or not np.isfinite(s):
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _h_from_coeffs(coeffs):
    """
    Convert DCT coefficients to step heights h.
    coeffs are the DCT-II coefficients of h.
    We use only the first K coefficients to restrict to low frequencies.
    """
    K = len(coeffs)
    full_coeffs = np.zeros(N)
    full_coeffs[:K] = coeffs
    # Inverse DCT-II
    h = idct(full_coeffs, type=2, norm='ortho')
    return h

def _objective_and_grad(coeffs, tau):
    """
    Compute smooth-max surrogate objective and gradient with respect to coeffs.
    """
    h = _h_from_coeffs(coeffs)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(coeffs)
    
    conv = np.convolve(h, h)
    
    # Smooth max approximation
    # soft_max(conv, tau) ≈ tau * log(sum(exp(conv_i / tau)))
    # For numerical stability, subtract max
    conv_max = np.max(conv)
    exp_conv = np.exp((conv - conv_max) / tau)
    log_sum_exp = conv_max + tau * np.log(np.sum(exp_conv))
    
    val = 2.0 * N * log_sum_exp / (s * s)
    val_abs = abs(val)
    
    # Gradient with respect to coeffs
    # We need d(val)/d(coeffs)
    # val = 2*N * softmax / s^2
    # d(softmax)/d(h[j]) = sum_k (conv_k's derivative wrt h[j]) * exp(conv_k/tau) / sum(exp(conv/tau))
    # d(conv_k)/d(h[j]) = h[k-j] + h[j-k] (with bounds checking)
    
    # This is expensive to compute analytically, so use finite differences for gradient
    # Actually, let's use a simpler approach: use scipy's numerical gradient or
    # just use the surrogate with a different optimization strategy.
    
    # For efficiency, let's compute gradient via the chain rule:
    # d(val)/d(coeffs) = d(val)/d(h) * d(h)/d(coeffs)
    # d(h)/d(coeffs) is the DCT matrix (orthogonal)
    
    # d(val)/d(h[j]) for val = 2*N * softmax / s^2:
    # = 2*N / s^2 * d(softmax)/d(h[j]) - 4*N*softmax/s^3 * 1
    
    # d(softmax)/d(h[j]) = sum_k w_k * d(conv_k)/d(h[j])
    # where w_k = exp(conv_k/tau) / sum(exp(conv/tau))
    # d(conv_k)/d(h[j]) = h[k-j] + h[j-k] (with h[i]=0 if i<0 or i>=N)
    
    w = exp_conv / np.sum(exp_conv)
    
    # Compute d(softmax)/d(h) efficiently
    # d(softmax)/d(h[j]) = sum_k w_k * (h[k-j] + h[j-k])
    # This is a convolution-like operation
    
    # For each j, sum_k w_k * h[k-j] + sum_k w_k * h[j-k]
    # = (w * h)[j] + (w * h)[N-1-j]? No, it's more like a cross-correlation
    
    # Let's just do it with a loop for clarity, N=400 is small
    d_softmax_dh = np.zeros(N)
    for j in range(N):
        val_j = 0.0
        for k in range(2*N - 1):
            # d(conv_k)/d(h[j])
            i1 = k - j
            i2 = j - k
            contrib = 0.0
            if 0 <= i1 < N:
                contrib += h[i1]
            if 0 <= i2 < N:
                contrib += h[i2]
            val_j += w[k] * contrib
        d_softmax_dh[j] = val_j
    
    d_val_dh = (2.0 * N / (s * s)) * d_softmax_dh - (4.0 * N * log_sum_exp / (s * s * s)) * np.ones(N)
    
    # d(val)/d(coeffs) = DCT_matrix.T @ d_val_dh (since h = DCT @ coeffs, and DCT is orthogonal)
    # h = idct(full_coeffs), so coeffs = dct(h)
    # d(h)/d(coeffs) is the DCT matrix, so d(coeffs)/d(h) is DCT.T
    # Since DCT is orthogonal, DCT.T = DCT^{-1}
    d_val_dcoeffs = dct(d_val_dh, type=2, norm='ortho')[:len(coeffs)]
    
    if val < 0:
        d_val_dcoeffs = -d_val_dcoeffs
    
    return val_abs, d_val_dcoeffs

def _objective(coeffs, tau):
    """Objective function for scipy minimize."""
    val, _ = _objective_and_grad(coeffs, tau)
    return val

def construct():
    """
    Construct step heights by optimizing DCT coefficients with adaptive smooth-max.
    """
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    best_h = np.ones(N)
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Use K low-frequency DCT coefficients
    K_list = [8, 12, 16, 20, 25, 30]
    
    # Multi-start with different K values and random initializations
    for K in K_list:
        for restart in range(3):
            if time.time() > deadline - 5:
                break
            
            # Initialize coeffs: mostly zero except a few random ones
            coeffs_init = np.zeros(K)
            # Add some structure: constant component + a few harmonics
            coeffs_init[0] = rng.uniform(0.5, 2.0)
            for k in range(1, min(K, 5)):
                coeffs_init[k] = rng.uniform(-0.5, 0.5)
            
            # Progressive tau reduction
            tau_schedule = [1.0, 0.5, 0.2, 0.1, 0.05]
            
            coeffs = coeffs_init.copy()
            for tau in tau_schedule:
                if time.time() > deadline - 5:
                    break
                
                # Optimize coeffs with fixed tau
                try:
                    def obj_t(c):
                        return _objective(c, tau)
                    
                    res = minimize(
                        obj_t,
                        coeffs,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
                    )
                    
                    if res.success and np.isfinite(res.fun):
                        coeffs = res.x
                except Exception:
                    pass
            
            # Evaluate final result
            h_candidate = _h_from_coeffs(coeffs)
            val_candidate = _evaluate(h_candidate)
            
            if np.isfinite(val_candidate) and val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
    
    # Also try a few full-random restarts with the smooth-max approach
    for K in [20, 30]:
        for restart in range(2):
            if time.time() > deadline - 5:
                break
            
            coeffs_init = rng.standard_normal(K) * 0.3
            coeffs_init[0] += 1.0  # Ensure non-zero sum
            
            tau_schedule = [1.0, 0.5, 0.2, 0.1]
            
            coeffs = coeffs_init.copy()
            for tau in tau_schedule:
                if time.time() > deadline - 5:
                    break
                
                try:
                    def obj_t(c):
                        return _objective(c, tau)
                    
                    res = minimize(
                        obj_t,
                        coeffs,
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False}
                    )
                    
                    if res.success and np.isfinite(res.fun):
                        coeffs = res.x
                except Exception:
                    pass
            
            h_candidate = _h_from_coeffs(coeffs)
            val_candidate = _evaluate(h_candidate)
            
            if np.isfinite(val_candidate) and val_candidate < best_val:
                best_h = h_candidate
                best_val = val_candidate
    
    # Final check
    final_val = _evaluate(best_h)
    if not np.isfinite(final_val) or final_val >= 2.0:
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
