"""Combined program: fast FFT screening + golden-section refinement + simulated annealing."""
import numpy as np
import time
import random

_START = time.time()
_DEADLINE = _START + 110.0  # Safety margin before 150s kill

N = 24
SQRTN = np.sqrt(N)

# Precompute indices for Horner evaluation
_KS = np.arange(N)

def _supnorm_grid(coeffs, n_grid=8192):
    """Fast sup-norm estimate using FFT on a grid."""
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    return float(np.max(np.abs(vals)))

def _eval_poly(coeffs, theta):
    """Evaluate |P(e^{i theta})| using vectorized dot product."""
    z = np.exp(1j * theta)
    return float(np.abs(np.dot(coeffs, z ** _KS)))

def _refine_peak(coeffs, ang, iters=60):
    """Golden-section refine max of |P(e^{i theta})| near ang."""
    a, b = ang - 0.01, ang + 0.01
    gr = (np.sqrt(5) - 1) / 2
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _eval_poly(coeffs, c)
    fd = _eval_poly(coeffs, d)
    
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _eval_poly(coeffs, c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _eval_poly(coeffs, d)
    
    return max(_eval_poly(coeffs, (a + b) / 2), fc, fd)

def _supnorm_precise(coeffs, n_grid=8192):
    """Sup-norm with golden-section refinement of top peaks."""
    padded = np.zeros(n_grid, dtype=float)
    padded[:N] = coeffs
    vals = np.fft.fft(padded)
    mags = np.abs(vals)
    
    # Find top peaks
    idx = np.argsort(mags)[-10:][::-1]
    best = 0.0
    for i in idx:
        ang = 2 * np.pi * i / n_grid
        v = _refine_peak(coeffs, ang)
        if v > best:
            best = v
    return best

def _gr_sequence(n):
    """Generate Golay-Rudin-Shapiro sequence of length n."""
    p, q = [1], [1]
    while len(p) < n:
        p, q = p + q, p + [-c for c in q]
    return p[:n]

def construct():
    # Generate initial Golay-Rudin-Shapiro sequence
    gr = np.array(_gr_sequence(N), dtype=float)
    
    # Multiple seed candidates
    rng = np.random.default_rng(42)
    seeds = [gr, -gr, np.ones(N), -np.ones(N)]
    
    # Add structured seeds
    for period in [3, 4, 5, 6, 7, 8]:
        seeds.append(np.array([1 if (k % period) < period // 2 else -1 for k in range(N)], dtype=float))
    
    # Add random seeds
    for _ in range(30):
        seeds.append(rng.choice([-1, 1], size=N).astype(float))
    
    best = None
    best_val = float('inf')
    
    # Evaluate seeds
    for s in seeds:
        val = _supnorm_grid(s, n_grid=4096)
        if val < best_val * 1.01:
            valr = _supnorm_precise(s)
            if valr < best_val:
                best_val = valr
                best = s.copy()
    
    # Simulated annealing with 1-2 bit flips
    cur = best.copy()
    cur_val = best_val
    T = 0.3
    
    while time.time() < _DEADLINE:
        # Flip 1 or 2 coefficients
        n_flips = rng.integers(1, 3)
        cand = cur.copy()
        for _ in range(n_flips):
            k = rng.integers(0, N)
            cand[k] *= -1
        
        # Fast screening
        val = _supnorm_grid(cand, n_grid=4096)
        
        if val < cur_val * 1.005:
            # Promising: do precise evaluation
            valr = _supnorm_precise(cand)
            if valr < cur_val:
                cur = cand
                cur_val = valr
                if valr < best_val:
                    best_val = valr
                    best = cand.copy()
            elif random.random() < np.exp(-(valr - cur_val) / T):
                cur = cand
                cur_val = valr
        else:
            # Accept worse with probability
            if random.random() < np.exp(-(val - cur_val) / T):
                cur = cand
                cur_val = val
        
        # Cool down
        T *= 0.9998
    
    # Final precise evaluation of best
    final_val = _supnorm_precise(best)
    if final_val < best_val:
        best_val = final_val
    
    return [int(c) for c in best]
