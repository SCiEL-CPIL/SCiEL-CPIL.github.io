"""Structural change: parameterize heights as a low-order cosine series and minimize the
normalized autoconvolution peak via a projected subgradient / Frank-Wolfe style search,
with a deadline guard."""
import numpy as np
import time

N = 600
HALF = N // 2  # 300

def _convolution_peak(h):
    # Full autoconvolution via FFT; peak is the maximum of the circular convolution
    # but since the function is on [-1/4, 1/4] and zero elsewhere, linear convolution
    # is what matters. Using FFT-based linear convolution.
    c = np.correlate(h, h, mode='full')
    return c.max()

def _normalized_peak(h):
    s = h.sum()
    if s <= 0:
        return float('inf')
    return _convolution_peak(h) / (s * s)

def construct():
    start = time.time()
    deadline = start + 110.0  # 10s safety margin

    # Initialize with a low-order cosine series
    # h[i] = a0 + sum_{k=1}^{K} a_k * cos(2*pi*k*i/N)
    # Use K=5 modes
    K = 5
    # Initialize coefficients
    # Start with a0 slightly above 0, and small oscillatory terms
    a = np.zeros(K + 1)
    a[0] = 0.5  # baseline
    for k in range(1, K + 1):
        a[k] = 0.05 * (k % 2 if k % 2 == 1 else -1)  # small alternating amplitudes

    def coeffs_to_heights(coeffs):
        i = np.arange(N)
        h = coeffs[0] * np.ones(N)
        for k in range(1, K + 1):
            h += coeffs[k] * np.cos(2 * np.pi * k * i / N)
        return np.maximum(h, 0.0)  # enforce nonnegativity

    best_h = coeffs_to_heights(a)
    best_score = _normalized_peak(best_h)

    # Simple iterative refinement: gradient-like steps in coefficient space
    # Use a random search with local perturbations to avoid getting stuck
    step_size = 0.05
    n_iter = 0

    while time.time() < deadline:
        n_iter += 1
        # Generate a perturbation
        da = np.random.normal(0, step_size, K + 1)
        # Keep a0 positive
        da[0] = abs(da[0])
        new_a = a + da
        new_h = coeffs_to_heights(new_a)
        new_score = _normalized_peak(new_h)

        if new_score < best_score:
            a = new_a
            best_h = new_h
            best_score = new_score
            # Reduce step size slightly on success
            step_size *= 0.95
        else:
            step_size *= 1.05  # increase if no progress
            step_size = min(step_size, 0.2)

        # Occasionally try a larger jump to escape local minima
        if n_iter % 50 == 0:
            jump_a = a.copy()
            jump_a[0] += 0.1
            jump_a[1:] += np.random.normal(0, 0.1, K)
            jump_h = coeffs_to_heights(jump_a)
            jump_score = _normalized_peak(jump_h)
            if jump_score < best_score:
                a = jump_a
                best_h = jump_h
                best_score = jump_score

    # Final check: also try a high-frequency alternating pattern as a fallback
    alt_h = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    alt_score = _normalized_peak(alt_h)
    if alt_score < best_score:
        best_h = alt_h

    return list(best_h)
