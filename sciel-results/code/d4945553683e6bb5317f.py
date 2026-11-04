"""Second autocorrelation inequality: maximize squared L2 norm / (L1 norm * sup norm of autoconvolution)."""
import time
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.signal import fftconvolve

N = 50
DEADLINE = 145.0  # safety margin before 150s kill

def _autoconvolution_peak(h):
    """Compute the supremum norm of the discrete autoconvolution of h.
    
    The autoconvolution is (h * h)(k) = sum_i h_i h_{i+k} for k = -(N-1)..(N-1).
    We compute this via FFT for efficiency.
    """
    # Full convolution of h with h (reversed for correlation, but autoconv is symmetric)
    # Actually, autoconvolution: (h h)(k) = sum_i h_i h_{i+k}
    # This is the same as np.correlate(h, h, mode='full')
    # Or equivalently, if we define g = h, then full convolution g * g gives sum_i g_i g_{j-i}
    # Let's use: autoconv[k] = sum_i h[i] h[i+k] for k = -(N-1), ..., 0, ..., N-1
    # Using FFT: pad to size >= 2*N-1
    size = 2 * N - 1
    h_pad = np.zeros(2 * size)
    h_pad[:N] = h
    H = np.fft.fft(h_pad, n=2 * size)
    auto = np.fft.ifft(H * H).real[:size]
    # The peak is the maximum absolute value
    return np.max(np.abs(auto))

def _compute_ratio(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    peak = _autoconvolution_peak(h)
    if peak < 1e-15:
        return 0.0
    return l2_sq / (l1 * peak)

def _objective_neg(h):
    """Negative of the ratio for minimization."""
    return -_compute_ratio(h)

def _smooth_max_obj(h, tau=1.0):
    """Smooth surrogate: use log-sum-exp for the max in autoconvolution."""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h ** 2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 1.0  # large positive (we minimize)
    
    # Compute autoconvolution
    size = 2 * N - 1
    h_pad = np.zeros(2 * size)
    h_pad[:N] = h
    H = np.fft.fft(h_pad, n=2 * size)
    auto = np.fft.ifft(H * H).real[:size]
    
    # Soft-max of |auto| values
    abs_auto = np.abs(auto)
    # log-sum-exp for numerical stability
    m = np.max(abs_auto)
    if m < 1e-30:
        return 1.0
    log_sum_exp = m + np.log(np.sum(np.exp(abs_auto - m)))
    smooth_peak = np.exp(log_sum_exp)  # This is actually sum(exp), not max... 
    
    # Actually for soft-max: (1/tau) * log(sum(exp(tau * x_i)))
    # Let's redo:
    tau_val = tau
    scaled = tau_val * abs_auto
    m2 = np.max(scaled)
    if m2 > 500:
        scaled = scaled - m2
    log_sum = m2 + np.log(np.sum(np.exp(scaled)))
    soft_max = log_sum / tau_val
    
    return l1 * soft_max / l2_sq  # We minimize this; equivalent to maximizing l2_sq/(l1*peak)

def _construct_gaussian(center, width):
    """Create a Gaussian-like profile."""
    x = np.linspace(0, N - 1, N)
    h = np.exp(-0.5 * ((x - center) / width) ** 2)
    # Ensure nonnegativity and normalize
    h = np.maximum(h, 0)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _construct_bimodal(c1, c2, w1, w2):
    """Create a bimodal profile."""
    x = np.linspace(0, N - 1, N)
    h1 = np.exp(-0.5 * ((x - c1) / w1) ** 2)
    h2 = np.exp(-0.5 * ((x - c2) / w2) ** 2)
    h = h1 + h2
    h = np.maximum(h, 0)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _construct_random_smooth(seed):
    """Create a random smooth profile using cosine basis."""
    rng = np.random.RandomState(seed)
    x = np.linspace(0, N - 1, N)
    # Use a few cosine modes
    n_modes = 3
    coeffs = rng.randn(n_modes)
    phases = rng.rand(n_modes) * 2 * np.pi
    h = np.ones(N)
    for m in range(n_modes):
        freq = (m + 1) * 2 * np.pi / N
        h += coeffs[m] * np.cos(freq * x + phases[m])
    h = np.maximum(h, 0)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _construct_triangular(center, width):
    """Triangular profile."""
    x = np.linspace(0, N - 1, N)
    h = np.maximum(1 - np.abs(x - center) / width, 0)
    if np.sum(h) > 0:
        h = h / np.sum(h)
    return h

def _optimize_from_start(h0, deadline):
    """Run L-BFGS-B from a starting point, respecting deadline."""
    h0 = np.asarray(h0, dtype=float).copy()
    
    # Bounds: h_i >= 0
    bounds = [(0, None)] * N
    
    best_h = h0.copy()
    best_val = _compute_ratio(h0)
    
    try:
        # First try with standard objective
        result = minimize(
            _objective_neg,
            h0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if result.success or result.fun < _objective_neg(best_h):
            val = _compute_ratio(result.x)
            if val > best_val:
                best_val = val
                best_h = result.x.copy()
        
        if time.time() > deadline:
            return best_h, best_val
        
        # Try smooth objective with different tau values
        for tau in [5.0, 10.0, 20.0, 50.0]:
            if time.time() > deadline:
                break
            try:
                result2 = minimize(
                    lambda h: _smooth_max_obj(h, tau=tau),
                    best_h,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
                )
                val2 = _compute_ratio(result2.x)
                if val2 > best_val:
                    best_val = val2
                    best_h = result2.x.copy()
            except Exception:
                pass
    except Exception:
        pass
    
    return best_h, best_val

def _polish(h, deadline):
    """High-precision polish pass with tighter tolerances."""
    h = np.asarray(h, dtype=float).copy()
    bounds = [(0, None)] * N
    
    try:
        result = minimize(
            _objective_neg,
            h,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
        )
        val = _compute_ratio(result.x)
        val_orig = _compute_ratio(h)
        if val > val_orig:
            return result.x, val
    except Exception:
        pass
    return h, _compute_ratio(h)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_h = None
    best_val = -1
    
    # Generate diverse initializations
    inits = []
    
    # Uniform (baseline)
    inits.append(np.ones(N) / N)
    
    # Gaussians with various centers and widths
    for center in [N * 0.2, N * 0.3, N * 0.5, N * 0.7, N * 0.8]:
        for width in [3, 5, 8, 12, 15]:
            inits.append(_construct_gaussian(center, width))
    
    # Bimodal
    for c1, c2 in [(N * 0.25, N * 0.75), (N * 0.3, N * 0.7), (N * 0.2, N * 0.8)]:
        for w in [5, 8, 10]:
            inits.append(_construct_bimodal(c1, c2, w, w))
    
    # Triangular
    for center in [N * 0.3, N * 0.5, N * 0.7]:
        for width in [10, 15, 20]:
            inits.append(_construct_triangular(center, width))
    
    # Random smooth profiles
    for seed in range(20):
        inits.append(_construct_random_smooth(seed))
    
    # Optimize from each initialization
    for i, h0 in enumerate(inits):
        if time.time() > deadline - 5:
            break
        h_opt, val = _optimize_from_start(h0, deadline - 5)
        if val > best_val:
            best_val = val
            best_h = h_opt.copy()
    
    # Polish the best result
    if best_h is not None and time.time() < deadline - 2:
        best_h, best_val = _polish(best_h, deadline - 1)
    
    if best_h is None:
        best_h = np.ones(N) / N
        best_val = _compute_ratio(best_h)
    
    # Ensure nonnegativity and return as list
    best_h = np.maximum(best_h, 0)
    return list(best_h)
