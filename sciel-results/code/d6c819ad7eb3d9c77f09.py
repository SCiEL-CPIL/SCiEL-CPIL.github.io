"""
Search for the flattest ±1 polynomial with 24 coefficients.

Strategy:
1. Start from the Golay-Rudin-Shapiro sequence (known to be a good starting point).
2. Use a fast FFT-based sup-norm estimator for quick evaluation during search.
3. Use precise golden-section refinement only for final scoring and critical decisions.
4. Multi-pass local search: single flips, pair flips, triple flips.
5. Simulated annealing escape to break out of local optima.
6. Time-budget aware: track elapsed time and return best found before deadline.
"""

import numpy as np
import time
from math import sqrt, pi, cos, sin

N_COEFFS = 24
TARGET_TIME = 145.0  # safety margin before 150s kill

# Precompute unit circle points for FFT-based evaluation
# Use a dense grid for fast sup-norm approximation
N_GRID = 4096
Z_GRID = np.exp(2j * pi * np.arange(N_GRID) / N_GRID)

def _rs_sequence(n):
    """Generate Golay-Rudin-Shapiro sequence of length n (padded/truncated)."""
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

def _fft_supnorm_fast(coeffs):
    """
    Fast sup-norm estimate using FFT on a dense grid.
    coeffs: list of 24 values in {+1, -1}
    Returns: (max_abs, argmax_index_in_grid)
    """
    # Pad coefficients to power of 2 >= N_GRID for efficient FFT
    # Actually, we evaluate P(z) = sum c_k z^k at z = exp(2*pi*i*j/N_GRID)
    # This is the DFT of the coefficient sequence (reversed or not depending on convention)
    # P(exp(2*pi*i*j/M)) = sum_{k=0}^{23} c_k exp(2*pi*i*j*k/M)
    # This is the M-point DFT of [c_0, c_1, ..., c_23, 0, ..., 0] evaluated at j=0..M-1
    # But we only have 24 coefficients, so we zero-pad to M points.
    
    m = N_GRID
    # Build signal: c_0, c_1, ..., c_23, 0, ..., 0 (length m)
    sig = np.zeros(m)
    sig[:N_COEFFS] = coeffs
    # DFT: X_j = sum_{k=0}^{m-1} sig_k exp(-2*pi*i*j*k/m)
    # But we want sum c_k exp(+2*pi*i*j*k/m) = conj(DFT of conj(coeffs))
    # Actually: sum c_k exp(2*pi*i*j*k/m) = DFT_j of the sequence where we flip sign in exponent
    # Let's just compute: for each j, P(z_j) = sum_k c_k * z_j^k
    # This is the inverse DFT convention. Let's use:
    # P(z_j) = IDFT of [c_0,...,c_23,0,...,0] at index j, scaled appropriately.
    # numpy.fft.ifft(X)[j] = (1/m) sum_k X_k exp(2*pi*i*j*k/m)
    # So if X = [c_0,...,c_23,0,...,0], then ifft(X)[j] * m = sum_k c_k exp(2*pi*i*j*k/m) = P(z_j)
    
    x = np.fft.ifft(sig) * m  # This gives P(z_j) for j = 0..m-1
    abs_vals = np.abs(x)
    max_idx = np.argmax(abs_vals)
    return abs_vals[max_idx], max_idx

def _precise_supnorm(coeffs, n_samples=8192):
    """
    More precise sup-norm using direct evaluation on a dense grid.
    Returns the maximum |P(z)| over |z|=1.
    """
    # Use a very dense grid
    angles = 2 * pi * np.arange(n_samples) / n_samples
    z = np.exp(1j * angles)
    # P(z) = sum c_k z^k
    # Vectorized: for each angle, compute sum
    # coeffs is shape (24,), z is shape (n_samples,)
    # We need z^k for k=0..23, then dot with coeffs
    powers = z[:, None] ** np.arange(N_COEFFS)[None, :]  # (n_samples, 24)
    vals = powers @ np.array(coeffs)
    return np.max(np.abs(vals))

def _refine_peak(coeffs, theta0, n_iter=50):
    """
    Golden-section refinement of the peak near theta0.
    Returns refined theta and |P(e^{i theta})|.
    """
    c_arr = np.array(coeffs)
    
    def p_abs(theta):
        z = np.exp(1j * theta)
        # P(z) = sum c_k z^k
        powers = z ** np.arange(N_COEFFS)
        return abs(np.sum(c_arr * powers))
    
    # Golden section search in a small interval around theta0
    gr = (5.0 - np.sqrt(5.0)) / 2.0  # golden ratio conjugate ~0.382
    width = 2 * pi / N_GRID  # initial width = one grid spacing
    a = theta0 - width / 2
    b = theta0 + width / 2
    
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = p_abs(c)
    fd = p_abs(d)
    
    for _ in range(n_iter):
        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = p_abs(d)
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = p_abs(c)
    
    theta = (a + b) / 2
    return theta, p_abs(theta)

def _score(coeffs):
    """
    Compute the score: max |P(z)| / sqrt(24) over |z|=1.
    Uses FFT for initial estimate, then refines the top peaks.
    """
    # Fast estimate
    max_val, max_idx = _fft_supnorm_fast(coeffs)
    theta0 = 2 * pi * max_idx / N_GRID
    
    # Refine the top peak
    _, refined_val = _refine_peak(coeffs, theta0, n_iter=60)
    
    # Also check a few neighboring peaks in case the FFT grid missed the true max
    # Check peaks at nearby indices
    best = refined_val
    for offset in [-1, 1, -2, 2]:
        idx2 = (max_idx + offset) % N_GRID
        theta2 = 2 * pi * idx2 / N_GRID
        _, v2 = _refine_peak(coeffs, theta2, n_iter=40)
        if v2 > best:
            best = v2
    
    return best / sqrt(N_COEFFS)

def _try_flip(coeffs, i):
    """Return new coeffs with coefficient i flipped."""
    new = list(coeffs)
    new[i] = -new[i]
    return new

def construct():
    start_time = time.time()
    
    # Initial solution: Golay-Rudin-Shapiro
    coeffs = _rs_sequence(N_COEFFS)
    best_coeffs = list(coeffs)
    best_score = _score(best_coeffs)
    
    def elapsed():
        return time.time() - start_time
    
    def time_remaining():
        return TARGET_TIME - elapsed()
    
    # Phase 1: Single-flip hill climbing (multiple passes)
    for pass_num in range(200):
        if time_remaining() < 5.0:
            break
        improved = False
        for i in range(N_COEFFS):
            if time_remaining() < 3.0:
                break
            new_coeffs = _try_flip(coeffs, i)
            new_score = _score(new_coeffs)
            if new_score < best_score - 1e-12:
                coeffs = new_coeffs
                best_coeffs = list(coeffs)
                best_score = new_score
                improved = True
        if not improved:
            break
    
    # Phase 2: Pair-flip search
    for pass_num in range(50):
        if time_remaining() < 5.0:
            break
        improved = False
        for i in range(N_COEFFS):
            if time_remaining() < 3.0:
                break
            for j in range(i + 1, N_COEFFS):
                if time_remaining() < 2.0:
                    break
                new_coeffs = list(coeffs)
                new_coeffs[i] = -new_coeffs[i]
                new_coeffs[j] = -new_coeffs[j]
                new_score = _score(new_coeffs)
                if new_score < best_score - 1e-12:
                    coeffs = new_coeffs
                    best_coeffs = list(coeffs)
                    best_score = new_score
                    improved = True
        if not improved:
            break
    
    # Phase 3: Triple-flip search
    for pass_num in range(20):
        if time_remaining() < 5.0:
            break
        improved = False
        for i in range(N_COEFFS):
            if time_remaining() < 3.0:
                break
            for j in range(i + 1, N_COEFFS):
                if time_remaining() < 2.0:
                    break
                for k in range(j + 1, N_COEFFS):
                    if time_remaining() < 1.5:
                        break
                    new_coeffs = list(coeffs)
                    new_coeffs[i] = -new_coeffs[i]
                    new_coeffs[j] = -new_coeffs[j]
                    new_coeffs[k] = -new_coeffs[k]
                    new_score = _score(new_coeffs)
                    if new_score < best_score - 1e-12:
                        coeffs = new_coeffs
                        best_coeffs = list(coeffs)
                        best_score = new_score
                        improved = True
        if not improved:
            break
    
    # Phase 4: Simulated annealing escape
    # Use a fast score for SA decisions, precise score only for tracking best
    T_start = 0.05
    T_end = 0.001
    sa_iters = 5000
    
    # Estimate how many iterations we can afford
    # Each iteration: one flip + fast score
    # Let's do a fixed number but check time
    current_coeffs = list(coeffs)
    current_score = best_score
    
    for it in range(sa_iters):
        if time_remaining() < 3.0:
            break
        
        # Cooling schedule
        progress = it / sa_iters
        T = T_start * (T_end / T_start) ** progress
        
        # Random flip
        i = np.random.randint(0, N_COEFFS)
        new_coeffs = _try_flip(current_coeffs, i)
        new_score = _score(new_coeffs)
        
        delta = new_score - current_score
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_coeffs = new_coeffs
            current_score = new_score
            if new_score < best_score:
                best_coeffs = list(new_coeffs)
                best_score = new_score
    
    # Final verification with precise scoring
    final_score = _score(best_coeffs)
    if final_score < best_score:
        best_score = final_score
    
    return best_coeffs

# Pre-seed random for reproducibility in SA
np.random.seed(42)
