import time
import numpy as np
from scipy.optimize import minimize

N = 400
_i = np.arange(N, dtype=float)
_t0 = time.time()
_DEADLINE = 140.0

def _ok():
    return (time.time() - _t0) < _DEADLINE

def _value(h):
    h = np.asarray(h, dtype=float)
    if not np.all(np.isfinite(h)):
        return np.inf
    s = float(np.sum(h))
    if s == 0.0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2.0 * N * float(np.max(conv)) / (s * s))

def _fft_conv(h):
    size = 2 * N
    H = np.fft.rfft(h, size)
    return np.fft.irfft(H * H, size)[:2 * N - 1]

def _smooth_max(conv, beta=50.0):
    m = np.max(conv)
    scaled = beta * (conv - m)
    scaled = np.clip(scaled, -100, 100)
    lse = m + np.log(np.sum(np.exp(scaled))) / beta
    return lse

def _obj(h):
    # We assume sum(h) is handled by constraint or normalization
    # To be safe, we compute the raw value but we will use constrained opt
    # For the optimizer, we want to minimize the max conv term
    # given that we will normalize h to sum 1 outside?
    # No, better to include the constraint in the optimizer.
    conv = _fft_conv(h)
    return _smooth_max(conv)

def construct():
    global _t0
    _t0 = time.time()
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N) / N
    best_v = _value(best_h)
    
    def update(h):
        nonlocal best_h, best_v
        h = np.asarray(h, dtype=float)
        if not np.all(np.isfinite(h)):
            return
        s = np.sum(h)
        if s == 0:
            return
        # Normalize to sum 1 for consistent comparison in objective
        h_norm = h / s
        v = _value(h_norm)
        if v < best_v:
            best_v = v
            best_h = h_norm.copy()

    # Define constraints for SLSQP
    # Constraint: sum(h) == 1
    cons = [{'type': 'eq', 'fun': lambda h: np.sum(h) - 1.0, 'jac': lambda h: np.ones(N)}]
    
    # Bounds: allow negative values, but keep them reasonable to avoid overflow
    # h values around 1/N = 0.0025.
    # Let's allow [-1, 1]
    bounds = [(-1.0, 1.0)] * N
    
    num_restarts = 20
    
    for r in range(num_restarts):
        if not _ok():
            break
        
        # Initialization
        if r == 0:
            h0 = np.ones(N) / N
        elif r == 1:
            # Gaussian window
            x = np.linspace(-1, 1, N)
            h0 = np.exp(-5 * x**2)
        elif r == 2:
            # Chirp
            h0 = np.cos(2 * np.pi * 10 * (np.arange(N) / N)**2)
        else:
            # Random
            h0 = rng.normal(0, 1, N)
            
        # Normalize initial guess to sum 1
        s0 = np.sum(h0)
        if s0 != 0:
            h0 = h0 / s0
        else:
            h0 = np.ones(N) / N
            
        try:
            res = minimize(
                _obj, 
                h0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=cons,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if res.x is not None:
                update(res.x)
        except Exception:
            pass

    # Polish phase
    if _ok():
        h_polish = best_h.copy()
        cur_v = _value(h_polish)
        step_size = 0.01
        max_iters = 50
        
        for it in range(max_iters):
            if not _ok():
                break
            improved = False
            order = rng.permutation(N)
            for idx in order:
                for d in (step_size, -step_size):
                    h2 = h_polish.copy()
                    h2[idx] += d
                    # Re-normalize to sum 1
                    s2 = np.sum(h2)
                    if s2 == 0: continue
                    h2 = h2 / s2
                    v2 = _value(h2)
                    if v2 < cur_v:
                        h_polish = h2
                        cur_v = v2
                        improved = True
                        break
            if not improved:
                step_size *= 0.7
                if step_size < 1e-5:
                    break
        update(h_polish)

    return best_h
