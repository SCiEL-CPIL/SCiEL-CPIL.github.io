"""Rewritten program: coordinate descent with line search for the second autocorrelation inequality.

Maximizes  C2 = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
for 50 nonnegative step heights.

Strategy:
  * Scale invariance -> fix ||h||_2 = 1 and minimize  F = ||h||_1 * ||h*h||_inf.
  * Use FFT for fast autoconvolution peak computation.
  * Coordinate descent with backtracking line search on each coordinate.
  * Multiple seed families: Gaussians, raised cosine, plateau, randomized.
"""
import time
import numpy as np
from scipy import signal

N = 50

# ---- fixed geometry -------------------------------------------------------
_dx = 1.0 / N


def _autoconv_peak_fft(h):
    """Return peak of linear autoconvolution h*h using FFT (O(N log N))."""
    # Pad to avoid circular convolution aliasing
    size = 2 * N - 1
    # Next power of 2 for FFT efficiency
    fft_size = 1 << (size - 1).bit_length()
    H = np.fft.rfft(h, n=fft_size)
    C = np.fft.irfft(H * H, n=fft_size)
    # Linear convolution is in C[0:size]
    c = C[:size]
    j = int(np.argmax(c))
    return float(c[j]), j


def _score(h):
    """C2 = ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    n2 = float(np.dot(h, h))
    if n2 <= 1e-300:
        return 0.0
    n1 = float(np.sum(h))
    peak, _ = _autoconv_peak_fft(h)
    if peak <= 1e-300 or n1 <= 1e-300:
        return 0.0
    return n2 / (n1 * peak)


def _objective(h):
    """F = ||h||_1 * ||h*h||_inf, to be minimized with ||h||_2 = 1."""
    n1 = float(np.sum(h))
    peak, _ = _autoconv_peak_fft(h)
    return n1 * peak


def _gaussian(alpha):
    """Gaussian-like nonnegative vector on the grid."""
    x = (np.arange(N) - (N - 1) / 2.0) * _dx
    v = np.exp(-alpha * x * x)
    return v


def _raised_cosine(width):
    """Raised cosine bump of given width (in grid units)."""
    x = (np.arange(N) - (N - 1) / 2.0)
    v = np.zeros(N)
    mask = np.abs(x) <= width
    v[mask] = 0.5 * (1 + np.cos(np.pi * x[mask] / width))
    return v


def _plateau(width, edge):
    """Plateau with smooth edges."""
    x = (np.arange(N) - (N - 1) / 2.0)
    v = np.zeros(N)
    inner = np.abs(x) <= width
    outer = (np.abs(x) > width) & (np.abs(x) <= width + edge)
    v[inner] = 1.0
    v[outer] = 0.5 * (1 + np.cos(np.pi * (np.abs(x[outer]) - width) / edge))
    return v


def _normalize2(h):
    """Normalize h to have ||h||_2 = 1."""
    n2 = float(np.dot(h, h))
    if n2 <= 1e-300:
        return h
    return h / np.sqrt(n2)


def _coordinate_descent(h0, deadline, max_iters):
    """Coordinate descent with line search to minimize F(h) = ||h||_1 * ||h*h||_inf
    subject to h >= 0 and ||h||_2 = 1.
    
    For each coordinate i, we optimize h_i while keeping others fixed,
    using a golden-section or bisection line search on [0, h_max].
    After each coordinate update, we renormalize to ||h||_2 = 1.
    """
    h = np.array(h0, dtype=float)
    h = np.maximum(h, 0.0)
    h = _normalize2(h)
    
    best_h = h.copy()
    best_F = _objective(h)
    
    t0 = time.time()
    it = 0
    no_improve_count = 0
    eps = 1e-10
    
    # Coordinate order: start with center, then outward (peak is likely near center)
    center = (N - 1) // 2
    order = list(range(center, -1, -1)) + list(range(center + 1, N))
    
    while it < max_iters:
        if time.time() > deadline:
            break
        
        F_old = _objective(h)
        improved = False
        
        for i in order:
            if time.time() > deadline:
                break
            
            h_old = h[i]
            if h_old < 1e-15:
                # Try small positive values
                candidates = [1e-4, 1e-3, 1e-2]
                best_cand_F = F_old
                best_cand_h = h_old
                for cv in candidates:
                    h[i] = cv
                    h = _normalize2(h)
                    F_cand = _objective(h)
                    if F_cand < best_cand_F - 1e-12:
                        best_cand_F = F_cand
                        best_cand_h = cv
                h[i] = best_cand_h
                h = _normalize2(h)
            else:
                # Line search: try scaling h_i by factors
                # The optimal h_i given other coords can be found by 1D search
                # F(h) = (sum_j h_j) * peak(h)
                # When we change h_i, both sum and peak change.
                # Use a simple multiplicative search: try h_i * r for r in a grid
                best_cand_F = F_old
                best_cand_h = h_old
                
                # Try a range of scaling factors
                for r in [0.1, 0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 10.0]:
                    new_val = h_old * r
                    if new_val < 0:
                        continue
                    h[i] = new_val
                    h = _normalize2(h)
                    F_cand = _objective(h)
                    if F_cand < best_cand_F - 1e-12:
                        best_cand_F = F_cand
                        best_cand_h = new_val
                
                # Refine around best
                if best_cand_h != h_old:
                    lo = max(0.0, best_cand_h * 0.7)
                    hi = best_cand_h * 1.3
                    for _ in range(5):
                        mid1 = lo + (hi - lo) / 3.0
                        mid2 = hi - (hi - lo) / 3.0
                        h[i] = mid1
                        h = _normalize2(h)
                        F1 = _objective(h)
                        h[i] = mid2
                        h = _normalize2(h)
                        F2 = _objective(h)
                        if F1 < F2:
                            hi = mid2
                        else:
                            lo = mid1
                    h[i] = (lo + hi) / 2.0
                    h = _normalize2(h)
                    F_cand = _objective(h)
                    if F_cand < best_cand_F - 1e-12:
                        best_cand_F = F_cand
                        best_cand_h = (lo + hi) / 2.0
                
                h[i] = best_cand_h
                h = _normalize2(h)
            
            if abs(h[i] - h_old) > eps:
                improved = True
        
        F_new = _objective(h)
        
        if F_new < best_F - 1e-12:
            best_F = F_new
            best_h = h.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 3:
                # Perturb to escape local minimum
                h = _normalize2(h + 1e-3 * np.random.randn(N))
                h = np.maximum(h, 0.0)
                h = _normalize2(h)
                no_improve_count = 0
        
        it += 1
    
    return best_h, best_F


def construct():
    """Return 50 nonnegative step heights maximizing C2.
    
    Runs coordinate descent from multiple seed families.
    """
    deadline = time.time() + 140.0  # safety margin
    rng = np.random.default_rng(42)
    
    best_h = None
    best_val = -1.0
    
    # Generate diverse seeds
    starts = []
    
    # Gaussian family
    for alpha in [10.0, 20.0, 40.0, 60.0, 100.0, 200.0, 400.0]:
        v = _gaussian(alpha)
        starts.append(v)
    
    # Raised cosine family
    for width in [5.0, 10.0, 15.0, 20.0, 25.0]:
        v = _raised_cosine(width)
        if np.sum(v) > 0:
            starts.append(v)
    
    # Plateau family
    for width, edge in [(5, 3), (10, 5), (15, 5), (20, 8)]:
        v = _plateau(width, edge)
        if np.sum(v) > 0:
            starts.append(v)
    
    # Randomized perturbations of best shapes
    for _ in range(6):
        base = _gaussian(rng.uniform(15.0, 150.0))
        v = base * (1.0 + 0.1 * rng.standard_normal(N))
        v = np.maximum(v, 0.0)
        starts.append(v)
    
    # Uniform (flat)
    starts.append(np.ones(N))
    
    # Sort seeds by initial objective (best first)
    for s in starts:
        s = _normalize2(s)
    
    starts.sort(key=lambda s: _objective(s))
    
    # Run coordinate descent on each seed with time budget
    n_starts = len(starts)
    for idx, s in enumerate(starts):
        if time.time() > deadline:
            break
        
        remaining = max(0.5, deadline - time.time())
        # Allocate more time to earlier (better) seeds
        weight = 1.0 / (1.0 + 0.3 * idx)
        total_weight = sum(1.0 / (1.0 + 0.3 * j) for j in range(n_starts - idx))
        per_time = remaining * (weight / total_weight)
        per_time = max(1.0, min(per_time, remaining))
        
        dl = time.time() + per_time
        max_iters = int(max(20, min(500, per_time * 50)))
        
        h, F = _coordinate_descent(s, dl, max_iters)
        val = _score(h)
        
        if val > best_val:
            best_val = val
            best_h = h.copy()
    
    if best_h is None:
        best_h = np.ones(N)
        best_val = _score(best_h)
    
    return [float(x) for x in best_h]


if __name__ == "__main__":
    t = time.time()
    res = construct()
    print("N =", len(res))
    print("score C2 =", _score(res))
    print("elapsed =", time.time() - t)
