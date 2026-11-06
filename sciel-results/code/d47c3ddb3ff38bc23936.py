import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import idct as scipy_idct

N = 400
_t0 = time.time()
_DEADLINE = 145.0  # Safety margin

_i_arr = np.arange(N)

def _value(h):
    """Compute the exact task value (lower is better)."""
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return np.inf
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = float(np.max(conv))
    val = abs(2.0 * N * m / (s * s))
    return val

def _smooth_value(h, beta=50.0):
    """
    Smooth surrogate for the objective using LogSumExp for max.
    """
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if abs(s) < 1e-12:
        return 1e10
    
    size = 2 * N
    H = np.fft.rfft(h, size)
    conv = np.fft.irfft(H * H, size)[:2 * N - 1]
    
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    
    val = abs(2.0 * N * lse / (s * s))
    return val

def construct():
    global _t0
    _t0 = time.time()
    
    best_h = np.ones(N)
    best_v = _value(best_h)
    
    def update_best(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_v:
            best_v = v
            best_h = h.copy()

    def deadline_ok():
        return (time.time() - _t0) < _DEADLINE

    rng = np.random.default_rng(42)
    
    # Initialization strategies
    inits = []
    inits.append(np.ones(N))
    inits.append(rng.normal(0, 1, N))
    inits.append(np.hamming(N))
    inits.append(np.hanning(N))
    inits.append(np.sin(2 * np.pi * _i_arr / N) + 1.0)
    inits.append(np.cos(2 * np.pi * _i_arr / N) + 1.0)
    
    # Add random DCT-based inits
    for _ in range(10):
        coeffs = rng.normal(0, 1, N)
        coeffs[0] = 10.0 # Ensure non-zero sum
        h_init = scipy_idct(coeffs, type=2, n=N, norm='ortho')
        inits.append(h_init)
        
    # Add some "block" inits
    for k in [10, 20, 50]:
        h_block = np.zeros(N)
        h_block[:k] = 1.0
        inits.append(h_block)

    # Optimization Loop
    for h0 in inits:
        if not deadline_ok():
            break
            
        s0 = np.sum(h0)
        if abs(s0) < 1e-6:
            continue
        h0_norm = h0 / s0
        
        try:
            # Phase 1: Coarse
            res = minimize(
                lambda h: _smooth_value(h, beta=20.0),
                h0_norm,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.x is not None and deadline_ok():
                h_candidate = res.x
                # Phase 2: Fine
                res2 = minimize(
                    lambda h: _smooth_value(h, beta=100.0),
                    h_candidate,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if res2.x is not None:
                    h_candidate = res2.x
                update_best(h_candidate)
        except Exception:
            pass

    # Final Polish: Coordinate Descent on the exact objective
    if deadline_ok() and best_v < np.inf:
        h_polish = best_h.copy()
        step = 0.05 * np.linalg.norm(h_polish)
        for _ in range(100):
            if not deadline_ok():
                break
            d = rng.normal(0, 1, N)
            h_try = h_polish + step * d
            v_try = _value(h_try)
            if v_try < best_v:
                h_polish = h_try
                best_h = h_polish.copy()
                best_v = v_try
                step *= 1.1 # Increase step if improved? Or keep constant
            else:
                h_try2 = h_polish - step * d
                v_try2 = _value(h_try2)
                if v_try2 < best_v:
                    h_polish = h_try2
                    best_h = h_polish.copy()
                    best_v = v_try2
                    step *= 1.1
                else:
                    step *= 0.5
                    if step < 1e-6:
                        break

    s = np.sum(best_h)
    if s == 0 or not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h

if __name__ == "__main__":
    h = construct()
    print(f"Value: {_value(h)}")
    print(f"Sum: {np.sum(h)}")
    print(f"Shape: {h.shape}")
