"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
import time

def _objective(h):
    """Compute the value to minimize."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    conv = np.convolve(h, h)
    mx = np.max(conv)
    val = abs(2.0 * N * mx / (s * s))
    return val

def _try_optimize(h0, max_iter=200, timeout=10.0):
    """Try to optimize from h0 using L-BFGS-B."""
    start = time.time()
    try:
        result = minimize(
            _objective, h0, method='L-BFGS-B',
            options={'maxiter': max_iter, 'maxfun': max_iter * 2}
        )
        if result.success or result.fun < _objective(h0):
            return result.x
    except Exception:
        pass
    return h0

def construct():
    deadline = time.time() + 110.0  # Safety margin before 120s kill
    best_h = np.ones(N)
    best_val = _objective(best_h)  # Should be 2.0
    
    # Try several analytical constructions
    candidates = []
    
    # 1. All ones (baseline)
    candidates.append(np.ones(N))
    
    # 2. Alternating signs
    candidates.append(np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]))
    
    # 3. Sine wave
    t = np.linspace(0, 4*np.pi, N)
    candidates.append(np.sin(t))
    
    # 4. Chirp
    t = np.linspace(0, 1, N)
    candidates.append(np.sin(2*np.pi*t**2*N))
    
    # 5. Random with zero mean
    rng = np.random.RandomState(42)
    candidates.append(rng.randn(N))
    
    # 6. Boxcar with negative center
    h6 = np.ones(N)
    h6[N//4:3*N//4] = -0.5
    candidates.append(h6)
    
    # 7. Two-bump positive/negative
    h7 = np.zeros(N)
    h7[:N//3] = 1.0
    h7[N//3:2*N//3] = -1.0
    h7[2*N//3:] = 1.0
    candidates.append(h7)
    
    # 8. Cosine windowed
    t = np.linspace(0, 2*np.pi, N)
    candidates.append(np.cos(t) * np.exp(-((t - np.pi)**2)/10))
    
    # 9. M-sequence-like (pseudo-random binary)
    mseq = np.zeros(N)
    x = 1
    for i in range(N):
        mseq[i] = 1.0 if x > 0 else -1.0
        # Simple LFSR-like update
        x = (x * 7 + 13) % 127
        if x == 0:
            x = 1
    candidates.append(mseq)
    
    # 10. Gaussian-like with negative tail
    t = np.linspace(-3, 3, N)
    candidates.append(np.exp(-t**2) - 0.3*np.exp(-(t-2)**2))
    
    # Evaluate all candidates
    for h in candidates:
        if not np.all(np.isfinite(h)):
            continue
        if abs(np.sum(h)) < 1e-12:
            continue
        val = _objective(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
        
        # Check deadline
        if time.time() > deadline - 30:
            break
    
    # Try optimization from the best few candidates
    # Sort candidates by objective value
    scored = []
    for h in candidates:
        if not np.all(np.isfinite(h)) or abs(np.sum(h)) < 1e-12:
            continue
        scored.append((_objective(h), h))
    scored.sort(key=lambda x: x[0])
    
    # Optimize from top 3 starting points
    for i, (val, h0) in enumerate(scored[:3]):
        if time.time() > deadline - 15:
            break
        h_opt = _try_optimize(h0, max_iter=100, timeout=8.0)
        if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-12:
            val_opt = _objective(h_opt)
            if val_opt < best_val:
                best_val = val_opt
                best_h = h_opt
    
    # One more round: try multi-start with random perturbations of best
    if time.time() < deadline - 20:
        rng2 = np.random.RandomState(123)
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            h_pert = best_h + 0.1 * rng2.randn(N)
            h_pert = np.sign(h_pert) * np.abs(h_pert)  # Keep sign structure? No, just use as is
            h_opt = _try_optimize(h_pert, max_iter=50, timeout=5.0)
            if np.all(np.isfinite(h_opt)) and abs(np.sum(h_opt)) > 1e-12:
                val_opt = _objective(h_opt)
                if val_opt < best_val:
                    best_val = val_opt
                    best_h = h_opt
    
    return best_h
# EVOLVE-BLOCK-END
