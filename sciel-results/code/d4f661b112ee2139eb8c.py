import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def _autoconv_max_abs(x):
    """Compute max absolute value of autoconvolution."""
    # Use FFT for efficient autoconvolution
    n = len(x)
    size = 2 * n - 1
    f = np.fft.rfft(x, size=size)
    ac = np.fft.irfft(f * np.conj(f), size=size)
    return np.max(np.abs(ac))

def _score(x):
    """Compute the objective: max_abs_autoconv / (sum x)^2."""
    s = np.sum(x)
    if s == 0:
        return np.inf
    return _autoconv_max_abs(x) / (s * s)

def _sign_flip_optimize(x, max_iter=2000):
    """Local search: flip signs to reduce the objective."""
    best = x.copy()
    best_score = _score(best)
    
    for it in range(max_iter):
        improved = False
        # Try flipping each sign
        for i in range(len(best)):
            old_val = best[i]
            best[i] = -old_val
            new_score = _score(best)
            if new_score < best_score - 1e-12:
                best_score = new_score
                improved = True
            else:
                best[i] = old_val
        if not improved:
            break
    return best, best_score

def _chirp_pattern(N):
    """Generate a chirp-like pattern with varying frequency."""
    x = np.ones(N)
    for i in range(N):
        phase = 2 * np.pi * i * i / (4 * N)
        x[i] = 1.0 if np.cos(phase) >= 0 else -1.0
    return x

def _block_pattern(N, block_size=10):
    """Generate alternating blocks."""
    x = np.ones(N)
    for start in range(0, N, 2 * block_size):
        end = min(start + block_size, N)
        x[start:end] = -1.0
    return x

def _random_pattern(N, seed):
    """Generate a random signed pattern."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1.0, 1.0], size=N)

def _structured_pattern(N):
    """Generate a structured pattern based on quadratic residues."""
    x = np.ones(N)
    for i in range(N):
        # Use a simple quadratic character
        q = (i * i) % 4
        if q == 1 or q == 3:
            x[i] = -1.0
    return x

def _sum_adjust(x, target_ratio=0.3):
    """Try to adjust the sum to be larger while keeping autoconvolution controlled.
    
    We can scale some entries or add a constant offset to increase the sum.
    """
    # Try adding a small constant to all entries
    best_x = x.copy()
    best_score = _score(best_x)
    
    for c in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, -0.1, -0.2, -0.3]:
        candidate = x + c
        # Clip to reasonable values
        candidate = np.clip(candidate, -10, 10)
        s = np.sum(candidate)
        if s == 0:
            continue
        sc = _score(candidate)
        if sc < best_score - 1e-12:
            best_score = sc
            best_x = candidate
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    best_x = np.ones(N)
    best_score = _score(best_x)
    
    # Generate diverse initial patterns
    patterns = [
        np.ones(N),
        _chirp_pattern(N),
        _block_pattern(N, 5),
        _block_pattern(N, 10),
        _block_pattern(N, 20),
        _structured_pattern(N),
    ]
    
    # Add some random patterns
    for seed in range(5):
        patterns.append(_random_pattern(N, seed))
    
    # Optimize each pattern
    for i, pattern in enumerate(patterns):
        if time.time() > deadline - 10.0:
            break
        
        # First do sign flip optimization
        x_opt, score_opt = _sign_flip_optimize(pattern, max_iter=1000)
        
        if score_opt < best_score - 1e-12:
            best_score = score_opt
            best_x = x_opt
        
        # Try sum adjustment
        if time.time() > deadline - 5.0:
            break
        x_adj, score_adj = _sum_adjust(x_opt)
        if score_adj < best_score - 1e-12:
            best_score = score_adj
            best_x = x_adj
        
        # Final sign flip optimization on adjusted
        if time.time() > deadline - 5.0:
            break
        x_final, score_final = _sign_flip_optimize(x_adj, max_iter=500)
        if score_final < best_score - 1e-12:
            best_score = score_final
            best_x = x_final
    
    return best_x
# EVOLVE-BLOCK-END
