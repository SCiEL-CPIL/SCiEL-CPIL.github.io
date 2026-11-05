"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autocorr_fft(h):
    """Compute full autocorrelation via FFT."""
    n = len(h)
    size = 2 * n
    H = np.fft.rfft(h, size)
    ac = np.fft.irfft(H * np.conj(H), size)
    # ac[0:n] is the positive lags (0 to n-1), ac[n:] wraps
    # Full conv: ac_full[i] for i=0..2n-2
    # ac[0] = sum(h^2), ac[1] = sum(h[k]*h[k+1]), etc.
    # ac[n-1] = sum(h[k]*h[k+n-1]) ... 
    # Actually irfft gives circular correlation of length size
    # For full linear autocorrelation:
    # positive lags: ac[0:n]
    # negative lags: ac[n:2n-1] reversed... 
    # Let me just use np.convolve for correctness, or construct properly
    # Actually for a real signal, the autocorrelation is symmetric
    # ac_full[k] = sum_j h[j] * h[j+k] for k >= 0
    # ac_full[-k] = ac_full[k]
    # From FFT: ac[0] = sum h^2, ac[1] = sum h[j]*h[j+1] (circular)
    # For linear: need to be careful. Let me just use np.convolve(h, h)
    # but that's O(N^2). With N=400, it's 160000 ops, fine.
    return np.convolve(h, h)

def _value(h):
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = np.convolve(h, h)
    return abs(2.0 * N * np.max(ac) / (s * s))

def _objective(x, tau=50.0):
    """Smooth surrogate using soft-max for the max in autocorrelation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = np.convolve(h, h)
    # soft-max approximation
    max_ac = np.max(ac)
    # Use log-sum-exp for smoothness
    tau_local = tau
    ac_shifted = ac - max_ac
    logsumexp = max_ac + np.log(np.sum(np.exp(tau_local * ac_shifted))) / tau_local
    val = 2.0 * N * logsumexp / (s * s)
    return val

def _objective_exact(x):
    h = x
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    ac = np.convolve(h, h)
    return 2.0 * N * np.max(ac) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    # Initial guess: try a few starting points
    best_h = np.ones(N)
    best_val = _value(best_h)
    
    # Try a few different initializations
    rng = np.random.RandomState(42)
    inits = []
    # Random
    inits.append(rng.randn(N))
    # Sine wave
    k = np.arange(N)
    inits.append(np.sin(2 * np.pi * 3 * k / N))
    inits.append(np.sin(2 * np.pi * 7 * k / N) + 0.5 * np.sin(2 * np.pi * 13 * k / N))
    # Barker-like: +1/-1
    inits.append(rng.choice([-1.0, 1.0], size=N))
    inits.append(np.sign(np.sin(2 * np.pi * 11 * k / N)))
    
    # Evaluate inits
    for h0 in inits:
        v = _value(h0)
        if v < best_val:
            best_val = v
            best_h = h0.copy()
    
    # Optimize from the best init
    h0 = best_h.copy()
    
    # Use L-BFGS-B with smooth surrogate
    result = None
    try:
        result = minimize(
            _objective, h0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            args=(50.0,)
        )
    except Exception:
        pass
    
    if result is not None and result.success:
        h_opt = result.x
        v = _value(h_opt)
        if v < best_val:
            best_val = v
            best_h = h_opt.copy()
    
    # Polish: use exact objective with more iterations
    elapsed = time.time() - start_time
    if elapsed < deadline - 10.0:
        try:
            result2 = minimize(
                _objective_exact, best_h,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.success:
                h_opt2 = result2.x
                v2 = _value(h_opt2)
                if v2 < best_val:
                    best_val = v2
                    best_h = h_opt2.copy()
        except Exception:
            pass
    
    # Final check
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:
        # Try one more from a different init
        h0b = rng.randn(N)
        h0b += 0.1 * np.sin(2 * np.pi * 5 * k / N)
        try:
            result3 = minimize(
                _objective, h0b,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10},
                args=(80.0,)
            )
            if result3.success:
                h_opt3 = result3.x
                v3 = _value(h_opt3)
                if v3 < best_val:
                    best_val = v3
                    best_h = h_opt3.copy()
        except Exception:
            pass
    
    # Ensure finite and sum not zero
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-12:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
