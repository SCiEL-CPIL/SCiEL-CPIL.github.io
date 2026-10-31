import numpy as np
import time
from scipy.optimize import minimize

N = 400
TIME_LIMIT = 140  # safety margin before 150s kill

def _autoconv(x):
    """Compute linear autoconvolution."""
    n = len(x)
    # Use FFT for efficiency
    nfft = 1
    while nfft < 2*n:
        nfft *= 2
    X = np.fft.rfft(x, nfft)
    C = np.fft.irfft(X * X, nfft)
    return C[:n]  # Only the first n values (lags 0 to n-1)

def _score(x):
    """Compute max|autoconv| / (sum x)^2."""
    S = np.sum(x)
    if abs(S) < 1e-12:
        return np.inf
    C = _autoconv(x)
    return np.max(np.abs(C)) / (S * S)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n, mapped to ±1."""
    tm = np.zeros(n, dtype=np.int8)
    for i in range(1, n):
        tm[i] = tm[i >> 1] ^ (i & 1)
    return 1 - 2*tm  # 0->1, 1->-1

def _random_sparse(n, density=0.5, seed=None):
    """Generate a random sparse signed sequence."""
    rng = np.random.default_rng(seed)
    mask = rng.random(n) < density
    signs = rng.choice([-1, 1], size=n)
    x = np.zeros(n)
    x[mask] = signs[mask]
    # Ensure non-zero sum
    if abs(np.sum(x)) < 1:
        x[mask[0]] = 1
    return x

def _local_optimize(x, max_iters=5000, time_budget=30):
    """Local sign-flip optimization."""
    start = time.time()
    best_score = _score(x)
    best_x = x.copy()
    n = len(x)
    
    # Precompute autoconv
    C = _autoconv(best_x)
    S = np.sum(best_x)
    
    for it in range(max_iters):
        if time.time() - start > time_budget:
            break
        improved = False
        # Random subset of positions to try
        rng = np.random.default_rng(it)
        try_positions = rng.choice(n, size=min(50, n), replace=False)
        
        for i in try_positions:
            # Try flipping sign
            x_new = best_x.copy()
            x_new[i] = -x_new[i]
            S_new = S - 2*x_new[i]
            if abs(S_new) < 1e-12:
                continue
            # Quick estimate: recompute autoconv
            C_new = _autoconv(x_new)
            score_new = np.max(np.abs(C_new)) / (S_new * S_new)
            if score_new < best_score - 1e-12:
                best_score = score_new
                best_x = x_new
                S = S_new
                C = C_new
                improved = True
                break
        
        if not improved:
            # Try small continuous adjustments
            for i in try_positions:
                for delta in [0.1, -0.1, 0.5, -0.5]:
                    x_new = best_x.copy()
                    x_new[i] += delta
                    S_new = S + delta
                    if abs(S_new) < 1e-12:
                        continue
                    C_new = _autoconv(x_new)
                    score_new = np.max(np.abs(C_new)) / (S_new * S_new)
                    if score_new < best_score - 1e-12:
                        best_score = score_new
                        best_x = x_new
                        S = S_new
                        C = C_new
                        improved = True
                        break
                if improved:
                    break
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_x = np.ones(N)
    best_score = _score(best_x)  # Should be 1/N
    
    # Try Thue-Morse
    if time.time() < deadline - 5:
        tm = _thue_morse(N)
        s = _score(tm)
        if s < best_score:
            best_score = s
            best_x = tm.copy()
    
    # Try random sparse initializations with local optimization
    rng = np.random.default_rng(42)
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        density = rng.uniform(0.3, 0.8)
        x0 = _random_sparse(N, density=density, seed=rng.integers(0, 2**32))
        
        x_opt, s = _local_optimize(x0, max_iters=2000, time_budget=5)
        if s < best_score:
            best_score = s
            best_x = x_opt
    
    # Final polish on best
    if time.time() < deadline - 5:
        remaining = deadline - 5 - time.time()
        if remaining > 2:
            x_final, s_final = _local_optimize(best_x, max_iters=5000, time_budget=remaining)
            if s_final < best_score:
                best_score = s_final
                best_x = x_final
    
    return best_x
