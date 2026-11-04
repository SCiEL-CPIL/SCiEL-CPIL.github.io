"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)
    return float(np.max(np.abs(conv[:2 * n - 1])))


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _soft_max_abs(x, beta=10.0):
    """Smooth approximation of max |autoconv| for gradient-based optimization."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    conv = np.fft.irfft(fx * fx, size)[:2 * n - 1]
    abs_conv = np.abs(conv)
    # Soft max using log-sum-exp for numerical stability
    m = np.max(abs_conv)
    if m < 1e-12:
        return m
    return m + np.log(np.sum(np.exp((abs_conv - m) * beta))) / beta


def _spectral_phase_optimize(deadline, rng, n_iter=5000, lr_init=0.01):
    """
    Optimize the phase of the DFT to minimize the soft max absolute autoconvolution,
    keeping the magnitude spectrum fixed (flat by default).
    """
    n = N
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    # Initialize with flat magnitude spectrum and random phases
    # Use rfft for real signal
    n_freq = size // 2 + 1
    magnitude = np.ones(n_freq)
    magnitude[0] = n  # DC component
    # For DC, phase is 0
    # For others, random phase
    phases = rng.uniform(0, 2 * np.pi, size=n_freq)
    phases[0] = 0.0
    
    # We'll optimize the complex spectrum X[k]
    # Start from random phases with flat magnitude
    X = magnitude * np.exp(1j * phases)
    
    # Compute initial x
    x = np.fft.irfft(X, size)
    x = x[:n]
    
    best_x = x.copy()
    best_val = _metric(best_x)
    
    lr = lr_init
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Compute autoconvolution
        fx_full = np.fft.rfft(x, size)
        conv_full = np.fft.irfft(fx_full * fx_full, size)
        conv = conv_full[:2 * n - 1]
        abs_conv = np.abs(conv)
        
        # Gradient of soft max
        m = np.max(abs_conv)
        if m < 1e-12:
            grad_conv = np.zeros_like(conv)
        else:
            weights = np.exp((abs_conv - m) * 10.0)
            weights /= np.sum(weights)
            # Sign of conv for gradient of |conv|
            signs = np.sign(conv)
            grad_conv = weights * signs
        
        # Gradient w.r.t. x
        # conv = x * x (linear)
        # d(conv[k])/dx[i] = x[k-i] + x[i-k] (with appropriate bounds)
        # For FFT-based, we can compute gradient in frequency domain
        # This is complex; let's use finite differences for robustness
        # Actually, let's use the frequency domain gradient
        
        # The objective is f(X) = soft_max(|ifft(FFT(x) * FFT(x))|)
        # Let's stick to time-domain gradient approximation via finite differences
        # This is expensive but robust
        
        # Actually, let's try a simpler approach:
        # Perturb the phases directly
        
        # Compute current value
        current_val = _metric(x)
        
        if current_val < best_val:
            best_val = current_val
            best_x = x.copy()
        
        # Perturb phases
        noise = rng.normal(0, lr, size=n_freq)
        noise[0] = 0  # Keep DC phase fixed
        new_phases = phases + noise
        X_new = magnitude * np.exp(1j * new_phases)
        x_new = np.fft.irfft(X_new, size)
        x_new = x_new[:n]
        
        new_val = _metric(x_new)
        
        if new_val < current_val:
            phases = new_phases
            X = X_new
            x = x_new
            # Small increase in lr
            lr = min(lr * 1.05, 0.1)
        else:
            # Decrease lr
            lr = max(lr * 0.95, 0.001)
    
    return best_x, best_val


def _try_local_search(x, deadline, rng, max_rounds=20000):
    """Local search with single flips and small perturbations."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Try flipping 1-3 random positions
        k = rng.integers(1, 4)
        indices = rng.choice(N, size=k, replace=False)
        x_flipped = best_x.copy()
        x_flipped[indices] *= -1
        
        v = _metric(x_flipped)
        if v < best_val:
            best_val = v
            best_x = x_flipped
    
    return best_x, best_val


def _try_bounded_values(deadline, rng):
    """Try sequences with values in a limited range, not just ±1."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Try different amplitude levels
    for amp in [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]:
        if time.time() > deadline:
            break
        for _ in range(20):
            cand = rng.choice([-amp, amp], size=N)
            v = _metric(cand)
            if v < best_val:
                best_val = v
                best_x = cand.copy()
    
    return best_x, best_val


def construct():
    deadline = time.time() + 130.0
    
    rng = np.random.default_rng(42)
    
    # Start from all-ones
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    # Phase 1: Spectral phase optimization
    if time.time() < deadline - 40:
        x_spec, val_spec = _spectral_phase_optimize(deadline - 40, rng, n_iter=8000)
        if val_spec < best_val:
            best_val = val_spec
            best_x = x_spec
    
    # Phase 2: Bounded value search
    if time.time() < deadline - 20:
        x_bnd, val_bnd = _try_bounded_values(deadline - 20, rng)
        if val_bnd < best_val:
            best_val = val_bnd
            best_x = x_bnd
    
    # Phase 3: Local search polish
    if time.time() < deadline - 5:
        best_x, best_val = _try_local_search(best_x, deadline - 5, rng, max_rounds=30000)
    
    return best_x
# EVOLVE-BLOCK-END
