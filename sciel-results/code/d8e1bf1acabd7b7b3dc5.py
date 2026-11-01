"""
Third autocorrelation inequality: minimize the upper bound on C3 for a signed step
function with 400 equal intervals on [-1/4, 1/4].

Strategy (Walsh-Hadamard spectral formulation):
  1. Represent heights h as a sparse linear combination of Walsh-Hadamard basis vectors.
  2. The autoconvolution in the time domain relates to the product of the Walsh spectrum.
  3. Minimize the L-infinity norm of the autoconvolution relative to the L1 norm squared.
  4. Use gradient descent on the sparse coefficients with a straight-through estimator
     for the sparsity pattern, followed by coordinate descent on the raw metric.
  5. Explicit deadline guard: return the best result found before the time limit.
"""

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _walsh_hadamard_transform(x):
    """
    Compute the Walsh-Hadamard transform of x using the Fast Walsh-Hadamard Transform (FWHT).
    This is an unnormalized transform: WH(x)_k = sum_j x_j * (-1)^{popcount(k & j)}
    """
    n = len(x)
    if n & (n - 1) != 0:
        raise ValueError("Length must be a power of 2")
    result = x.copy().astype(np.float64)
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                a = result[j]
                b = result[j + h]
                result[j] = a + b
                result[j + h] = a - b
        h *= 2
    return result


def _inverse_walsh_hadamard_transform(y):
    """
    Compute the inverse Walsh-Hadamard transform. Since WH is its own inverse up to a factor,
    WH^{-1}(y) = WH(y) / n.
    """
    n = len(y)
    return _walsh_hadamard_transform(y) / n


def _build_walsh_basis(n, max_order=64):
    """
    Build a set of Walsh-Hadamard basis vectors up to a certain order.
    We use the first `max_order` basis vectors (excluding the all-ones vector which is index 0).
    """
    # We need n to be a power of 2 for FWHT to work cleanly
    # N=400 is not a power of 2, so we'll pad to 512 and truncate
    n_pad = 1 << (n - 1).bit_length()
    basis = []
    for k in range(1, max_order + 1):
        # Generate the k-th Walsh-Hadamard sequence
        seq = np.zeros(n_pad)
        for j in range(n_pad):
            seq[j] = 1.0 if (np.bitwise_count(k & j) % 2 == 0) else -1.0
        basis.append(seq[:n])
    return np.array(basis)


def _raw_metric(h):
    """
    Compute the raw C3 upper-bound metric for a height vector h:
        max_k |conv(h, h)_k| / (sum_i |h_i|)^2
    Uses FFT for the linear convolution.
    """
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    conv = C[:2 * N - 1]
    denom = np.sum(np.abs(h)) ** 2
    if denom < 1e-30:
        return np.inf
    return np.max(np.abs(conv)) / denom


def _walsh_objective(coeffs, basis, T=1.0, eps=1e-10):
    """
    Objective function for optimization over Walsh coefficients.
    h = basis @ coeffs
    Minimize: max_k |conv(h, h)_k| / (sum_i |h_i|)^2
    """
    h = basis @ coeffs
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    H = np.fft.rfft(h, n=n_fft)
    C = np.fft.irfft(H * H, n=n_fft)
    conv = C[:2 * N - 1]
    
    abs_conv = np.sqrt(conv ** 2 + eps)
    max_abs = np.max(abs_conv)
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    smooth_max = T * log_sum_exp
    
    sum_abs_h = np.sum(np.sqrt(h ** 2 + eps))
    return smooth_max / (sum_abs_h ** 2)


def _coordinate_polish_raw(h, max_sweeps=100, time_budget=30.0):
    """
    Coordinate descent on the raw metric, trying to flip each coordinate to +1 or -1.
    """
    h = h.copy()
    current = _raw_metric(h)
    if not np.isfinite(current):
        return h
    
    start_time = time.time()
    
    for sweep in range(max_sweeps):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        improved = False
        for i in range(N):
            elapsed = time.time() - start_time
            if elapsed > time_budget:
                break
            
            old = h[i]
            
            # Try +1
            h[i] = 1.0
            v = _raw_metric(h)
            if np.isfinite(v) and v < current - 1e-12:
                current = v
                improved = True
                continue
            
            # Try -1
            h[i] = -1.0
            v = _raw_metric(h)
            if np.isfinite(v) and v < current - 1e-12:
                current = v
                improved = True
                continue
            
            # Restore
            h[i] = old
        
        if not improved:
            break
    
    return h


def construct():
    """
    Main entry point. Returns a sequence of signed step heights.
    """
    start_time = time.time()
    deadline = 145.0  # 5s margin under 150s
    
    # Build Walsh-Hadamard basis (use 512-dimensional padded basis, truncated to 400)
    n_pad = 512
    max_order = 64
    basis_full = []
    for k in range(1, max_order + 1):
        seq = np.zeros(n_pad)
        for j in range(n_pad):
            seq[j] = 1.0 if (np.bitwise_count(k & j) % 2 == 0) else -1.0
        basis_full.append(seq[:N])
    basis = np.array(basis_full)
    
    # Initialize with random sparse coefficients
    rng = np.random.default_rng(123)
    
    best_h = np.ones(N)
    best_val = _raw_metric(best_h)
    
    # Multiple restarts with different sparsity patterns
    n_restarts = 8
    for r in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        # Create sparse initial coefficients
        coeffs = np.zeros(max_order)
        if r == 0:
            # Start with all coefficients small
            coeffs[:] = 0.1
        elif r == 1:
            # Random signs for first 16 coefficients
            coeffs[:16] = rng.choice([-1, 1], size=16) * rng.uniform(0.5, 1.5, size=16)
        elif r == 2:
            # Sparse: only 8 coefficients active
            active = rng.choice(max_order, size=8, replace=False)
            coeffs[active] = rng.choice([-1, 1], size=8) * rng.uniform(0.5, 2.0, size=8)
        elif r == 3:
            # Even sparser: 4 coefficients
            active = rng.choice(max_order, size=4, replace=False)
            coeffs[active] = rng.choice([-1, 1], size=4) * rng.uniform(1.0, 3.0, size=4)
        elif r == 4:
            # Dense but small
            coeffs[:] = rng.normal(0, 0.3, size=max_order)
        elif r == 5:
            # Block structure in Walsh domain
            coeffs[:32] = rng.choice([-1, 1], size=32) * rng.uniform(0.5, 1.0, size=32)
            coeffs[32:] = 0
        elif r == 6:
            # Alternating blocks
            for b in range(max_order // 8):
                coeffs[b*8:(b+1)*8] = (1 if b % 2 == 0 else -1) * rng.uniform(0.5, 1.5)
        else:
            # Random
            coeffs[:] = rng.normal(0, 0.5, size=max_order)
        
        try:
            # Optimize coefficients using L-BFGS-B
            res = minimize(
                lambda c: _walsh_objective(c, basis),
                coeffs,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            h_cand = basis @ res.x
            val = _raw_metric(h_cand)
            
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_h = h_cand
                
        except Exception:
            pass
    
    # Coordinate descent polish on the best result
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        time_for_polish = min(40.0, deadline - 10 - elapsed)
        best_h = _coordinate_polish_raw(best_h, max_sweeps=50, time_budget=time_for_polish)
        best_val = _raw_metric(best_h)
    
    # Final check
    if not np.isfinite(best_val):
        best_h = np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
