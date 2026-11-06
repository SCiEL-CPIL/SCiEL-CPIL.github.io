"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_t0 = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill

def _value(h):
    """Compute the exact objective value."""
    h = np.asarray(h, dtype=float)
    if len(h) != N or not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    m = float(np.max(conv))
    return abs(2.0 * N * m / (s * s))

def _fft_convolve_max(h):
    """Compute max of full autocorrelation using FFT (faster)."""
    n = len(h)
    size = 2 * n
    f = np.fft.rfft(h, size)
    conv = np.fft.irfft(f * np.conj(f), size)[:n]
    return float(np.max(conv))

def _fast_value(h):
    """Fast objective using FFT-based convolution."""
    s = float(np.sum(h))
    if s == 0.0 or not np.all(np.isfinite(h)):
        return np.inf
    m = _fft_convolve_max(h)
    return abs(2.0 * N * m / (s * s))

def construct():
    global _t0
    _t0 = time.time()
    
    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if len(h) != N or not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if abs(s) < 1e-15:
            return
        v = _value(h)
        if v < best_v - 1e-12:
            best_v = v
            best_h = h.copy()
    
    # Strategy: Use low-PAPR sequences (like Zadoff-Chu) as base
    # Add small DC offset to make sum non-zero
    # The key insight: if h has small peak autocorrelation relative to its energy,
    # and we add a small constant to make sum != 0, we can get V < 2
    
    # Zadoff-Chu sequence generation
    def zadoff_chu(N, root):
        """Generate Zadoff-Chu sequence of length N with given root."""
        n = np.arange(N)
        return np.exp(-1j * np.pi * root * n * (n + 1) / N)
    
    # For real vectors, we can use the real part of complex Zadoff-Chu
    # or use Golay complementary pairs
    
    # Try different roots and approaches
    for root in [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
        if not deadline_ok():
            break
        
        # Get real part of Zadoff-Chu
        zc = zadoff_chu(N, root)
        h_base = zc.real
        
        # Normalize and try different DC offsets
        h_base = h_base / np.max(np.abs(h_base))
        
        # Try a range of DC offsets
        for c in np.linspace(-0.5, 0.5, 21):
            h = h_base + c
            if abs(np.sum(h)) < 1e-10:
                continue
            v = _fast_value(h)
            if v < best_v:
                update_best(h)
    
    # Also try using Golay complementary pairs
    # Golay sequences have the property that their autocorrelation sums cancel
    # For length 400, we can use recursive construction or look for good pairs
    
    # Simpler: use random vectors with optimization
    rng = np.random.default_rng(42)
    
    for attempt in range(10):
        if not deadline_ok():
            break
        
        # Generate random vector with some structure
        # Mix of low-frequency and high-frequency components
        h = np.zeros(N)
        
        # Add some low-frequency sinusoids
        for k in range(1, 5):
            amp = rng.random()
            freq = k * rng.random() * 2
            h += amp * np.sin(2 * np.pi * freq * np.arange(N) / N + rng.random() * 2 * np.pi)
        
        # Add high-frequency noise
        h += 0.1 * rng.standard_normal(N)
        
        # Try to optimize with gradient-based method
        def obj(params):
            # params[0] is DC offset, rest are sinusoid amplitudes
            c = params[0]
            h_trial = h.copy()
            for k in range(1, min(5, len(params) - 1)):
                h_trial += params[k] * np.sin(2 * np.pi * k * np.arange(N) / N)
            return _fast_value(h_trial)
        
        init = np.zeros(5)
        init[0] = 0.1  # Small DC offset
        
        try:
            res = minimize(
                obj,
                init,
                method='L-BFGS-B',
                bounds=[(-1, 1)] + [(-1, 1)] * 4,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if res.x is not None:
                c = res.x[0]
                h_trial = h.copy()
                for k in range(1, 5):
                    h_trial += res.x[k] * np.sin(2 * np.pi * k * np.arange(N) / N)
                if abs(np.sum(h_trial)) > 1e-10:
                    update_best(h_trial)
        except Exception:
            pass
    
    # Final validation
    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
    
    v_final = _value(best_h)
    if v_final < 2.0 - 1e-12:
        return best_h
    else:
        return np.ones(N)
# EVOLVE-BLOCK-END
