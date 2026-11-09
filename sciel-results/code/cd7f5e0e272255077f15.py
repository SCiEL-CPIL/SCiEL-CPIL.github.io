"""Spectral optimization for Erdos minimum overlap."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 140.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _start_time

def _deadline_ok(margin=2.0):
    return _elapsed() < _TIME_LIMIT - margin

def _max_overlap(h):
    """Compute max over circular shifts of sum_i h_i * (1 - h_{(i+k)%N})."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    min_autocorr = np.min(autocorr)
    return N / 2.0 - min_autocorr

def _fix_sum(h):
    """Adjust h to have sum exactly N/2, keeping values in [0,1]."""
    target = N / 2.0
    current_sum = np.sum(h)
    diff = target - current_sum
    h = h + diff / N
    h = np.clip(h, 0.0, 1.0)
    for _ in range(10):
        current_sum = np.sum(h)
        diff = target - current_sum
        if abs(diff) < 1e-12:
            break
        h = h + diff / N
        h = np.clip(h, 0.0, 1.0)
    return h

def _spectral_optimize():
    """
    Model h_i = 0.5 + a * cos(2*pi*k*i/N + phi) for various frequencies k.
    
    The overlap at shift s is:
    overlap(s) = sum_i h_i * (1 - h_{(i+s)%N})
               = N/2 - sum_i h_i * h_{(i+s)%N}
    
    For h_i = 0.5 + a*cos(2*pi*k*i/N + phi):
    autocorr(s) = sum_i h_i * h_{(i+s)%N}
                = N/4 + a^2/2 * cos(2*pi*k*s/N)
    
    So overlap(s) = N/2 - N/4 - a^2/2 * cos(2*pi*k*s/N)
                  = N/4 - a^2/2 * cos(2*pi*k*s/N)
    
    Max overlap = N/4 + a^2/2 (when cos = -1)
    Min overlap = N/4 - a^2/2 (when cos = +1)
    
    To minimize max overlap, we want to minimize a^2/2, i.e., a=0.
    But wait - this means the constant solution is optimal for single frequency!
    
    However, with multiple frequencies or non-sinusoidal patterns, we can do better.
    Let's try: h_i = 0.5 + sum_m a_m * cos(2*pi*m*i/N + phi_m)
    
    For the sum of cosines, the autocorrelation becomes more complex.
    Let's just try various patterns and optimize.
    """
    best_h = np.full(N, 0.5)
    best_overlap = _max_overlap(best_h)
    
    # Try various frequency patterns
    for k in range(1, N // 2 + 1):
        if not _deadline_ok(5.0):
            break
        for a in [0.1, 0.2, 0.3, 0.4, 0.49]:
            for phi in np.linspace(0, 2 * np.pi, 20, endpoint=False):
                h = 0.5 + a * np.cos(2 * np.pi * k * np.arange(N) / N + phi)
                h = _fix_sum(h)
                ov = _max_overlap(h)
                if ov < best_overlap:
                    best_overlap = ov
                    best_h = h.copy()
    
    return best_h, best_overlap

def _try_bipartite():
    """Try alternating high/low pattern."""
    best_h = np.full(N, 0.5)
    best_overlap = _max_overlap(best_h)
    
    # Try alternating pattern with different amplitudes
    for a in np.linspace(0.1, 0.49, 10):
        h = np.zeros(N)
        h[0::2] = 0.5 + a
        h[1::2] = 0.5 - a
        h = _fix_sum(h)
        ov = _max_overlap(h)
        if ov < best_overlap:
            best_overlap = ov
            best_h = h.copy()
    
    return best_h, best_overlap

def _try_block_patterns():
    """Try block patterns: consecutive high/low regions."""
    best_h = np.full(N, 0.5)
    best_overlap = _max_overlap(best_h)
    
    for block_size in range(1, N // 2 + 1):
        if not _deadline_ok(5.0):
            break
        for a in [0.1, 0.2, 0.3, 0.4, 0.49]:
            h = np.zeros(N)
            # Create blocks of high and low
            num_blocks = N // (2 * block_size)
            for b in range(num_blocks):
                start = b * 2 * block_size
                h[start:start + block_size] = 0.5 + a
                h[start + block_size:start + 2 * block_size] = 0.5 - a
            # Handle remainder
            remaining = N - num_blocks * 2 * block_size
            if remaining > 0:
                h[num_blocks * 2 * block_size:] = 0.5
            h = _fix_sum(h)
            ov = _max_overlap(h)
            if ov < best_overlap:
                best_overlap = ov
                best_h = h.copy()
    
    return best_h, best_overlap

def _local_search(h, max_iters=3000):
    """Coordinate descent / local search to improve the solution."""
    h = _fix_sum(h.copy())
    best_overlap = _max_overlap(h)
    
    for it in range(max_iters):
        if not _deadline_ok(5.0):
            break
        improved = False
        # Try adjusting pairs of variables
        for i in range(N):
            if not _deadline_ok(5.0):
                break
            for j in range(i + 1, min(i + 20, N)):  # Only nearby pairs for speed
                for delta in [0.01, 0.05, 0.1]:
                    h_new = h.copy()
                    h_new[i] -= delta
                    h_new[j] += delta
                    if 0 <= h_new[i] <= 1 and 0 <= h_new[j] <= 1:
                        h_new = _fix_sum(h_new)
                        ov = _max_overlap(h_new)
                        if ov < best_overlap - 1e-10:
                            h = h_new
                            best_overlap = ov
                            improved = True
                            break
                    h_new = h.copy()
                    h_new[i] += delta
                    h_new[j] -= delta
                    if 0 <= h_new[i] <= 1 and 0 <= h_new[j] <= 1:
                        h_new = _fix_sum(h_new)
                        ov = _max_overlap(h_new)
                        if ov < best_overlap - 1e-10:
                            h = h_new
                            best_overlap = ov
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return h, best_overlap

def construct():
    # Start with constant solution
    best_h = np.full(N, 0.5)
    best_overlap = _max_overlap(best_h)
    
    # Try spectral optimization (single frequency sinusoidal perturbations)
    h1, ov1 = _spectral_optimize()
    if ov1 < best_overlap:
        best_overlap = ov1
        best_h = h1
    
    # Try bipartite (alternating) patterns
    h2, ov2 = _try_bipartite()
    if ov2 < best_overlap:
        best_overlap = ov2
        best_h = h2
    
    # Try block patterns
    h3, ov3 = _try_block_patterns()
    if ov3 < best_overlap:
        best_overlap = ov3
        best_h = h3
    
    # Local search polish on the best candidate
    if _deadline_ok(5.0):
        h_best, ov_best = _local_search(best_h, max_iters=2000)
        if ov_best < best_overlap:
            best_overlap = ov_best
            best_h = h_best
    
    best_h = _fix_sum(best_h)
    return best_h
# EVOLVE-BLOCK-END
