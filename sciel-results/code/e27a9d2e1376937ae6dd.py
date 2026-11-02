"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 600

def _score_step(h):
    """Compute the score for a step function h (array of length N).
    Score = max(autoconv) / (sum(h))^2
    """
    if np.any(h < 0):
        return float('inf')
    s = h.sum()
    if s <= 0:
        return float('inf')
    # Compute autoconvolution via FFT
    # Linear convolution length is 2*N - 1
    size = 2 * N
    F = np.fft.rfft(h, size)
    auto = np.fft.irfft(F * F, size)[:2 * N - 1]
    peak = np.max(auto)
    return peak / (s * s)

def _make_gaussian(sigma_frac, amplitude):
    """Create a Gaussian step function."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = amplitude * np.exp(-0.5 * (x / sigma_frac) ** 2)
    # Ensure non-negative
    h = np.maximum(h, 0)
    # Normalize so that sum(h) = 1 (so score = max(auto))
    if h.sum() > 0:
        h = h / h.sum()
    else:
        h = np.ones(N) / N
    return h

def _make_triangular(width_frac, amplitude):
    """Create a triangular step function."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.maximum(1 - np.abs(x) / width_frac, 0) * amplitude
    h = np.maximum(h, 0)
    if h.sum() > 0:
        h = h / h.sum()
    else:
        h = np.ones(N) / N
    return h

def _make_box(width_frac):
    """Create a box step function with given width."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    h = np.where(np.abs(x) <= width_frac, 1.0, 0.0)
    if h.sum() > 0:
        h = h / h.sum()
    else:
        h = np.ones(N) / N
    return h

def _make_bump(p, width_frac):
    """Create a polynomial bump function of degree p."""
    x = np.linspace(-0.25, 0.25, N, endpoint=False)
    u = np.abs(x) / width_frac
    h = np.where(u < 1, (1 - u ** 2) ** p, 0)
    h = np.maximum(h, 0)
    if h.sum() > 0:
        h = h / h.sum()
    else:
        h = np.ones(N) / N
    return h

def _optimize_gaussian(sigma_frac):
    """For a given sigma, the optimal amplitude scaling is determined by normalization.
    But the shape (sigma) affects the peak of autoconv.
    For a Gaussian, autoconv is also Gaussian with variance 2*sigma^2.
    Peak autoconv = (sum h)^2 * (1/(2*pi*2*sigma^2 * dx^2)) approx...
    Actually, for discrete Gaussian, we can just compute it.
    """
    h = _make_gaussian(sigma_frac, 1.0)
    return _score_step(h)

def _optimize_triangle(width_frac):
    h = _make_triangular(width_frac, 1.0)
    return _score_step(h)

def _optimize_box(width_frac):
    h = _make_box(width_frac)
    return _score_step(h)

def _optimize_bump(p, width_frac):
    h = _make_bump(p, width_frac)
    return _score_step(h)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Leave 20s safety margin
    
    best_score = float('inf')
    best_h = np.ones(N) / N
    
    # Update with box (width = full interval)
    h_box = _make_box(0.25)
    s = _score_step(h_box)
    if s < best_score:
        best_score = s
        best_h = h_box.copy()
    
    # Search over Gaussian sigmas
    for sigma_frac in np.linspace(0.01, 0.25, 50):
        if time.time() > deadline:
            break
        h = _make_gaussian(sigma_frac, 1.0)
        s = _score_step(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Search over triangular widths
    for width_frac in np.linspace(0.01, 0.25, 50):
        if time.time() > deadline:
            break
        h = _make_triangular(width_frac, 1.0)
        s = _score_step(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Search over box widths
    for width_frac in np.linspace(0.01, 0.25, 50):
        if time.time() > deadline:
            break
        h = _make_box(width_frac)
        s = _score_step(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # Search over polynomial bumps
    for p in [1, 2, 3, 4]:
        for width_frac in np.linspace(0.01, 0.25, 30):
            if time.time() > deadline:
                break
            h = _make_bump(p, width_frac)
            s = _score_step(h)
            if s < best_score:
                best_score = s
                best_h = h.copy()
    
    # Refine: local search around best found
    if best_score < float('inf'):
        # Try small perturbations
        for _ in range(50):
            if time.time() > deadline:
                break
            # Random smooth perturbation
            noise = np.random.randn(N)
            # Smooth the noise
            kernel = np.exp(-np.arange(-20, 21) ** 2 / 10)
            noise = np.convolve(noise, kernel, mode='same')
            noise = (noise - noise.mean()) * 0.01
            h_pert = best_h + noise
            h_pert = np.maximum(h_pert, 0)
            if h_pert.sum() <= 0:
                continue
            h_pert = h_pert / h_pert.sum()
            s = _score_step(h_pert)
            if s < best_score:
                best_score = s
                best_h = h_pert.copy()
    
    # Final check: ensure non-negative and return list
    best_h = np.maximum(best_h, 0)
    if best_h.sum() <= 0:
        best_h = np.ones(N) / N
    else:
        # Don't normalize - the task says "sequence of nonnegative step heights"
        # The score is normalized by (sum)^2, so scaling doesn't matter.
        # But to be safe, keep the heights as they are.
        pass
    
    return list(best_h)

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
