"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _autoconv(x):
    """Compute autoconvolution via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    F = np.fft.rfft(x, size)
    ac = np.fft.irfft(F * F.conj(), size)[:2*n-1]
    return ac

def _max_abs_ac(x):
    """Maximum absolute value of autoconvolution."""
    ac = _autoconv(x)
    return np.max(np.abs(ac))

def _score(x):
    """Score: max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-10:
        return 1e10
    return _max_abs_ac(x) / (s * s)

def _random_signed(n, seed=None):
    """Generate random ±1 sequence."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    return rng.choice([-1.0, 1.0], size=n)

def _local_search(x, max_iter=2000, deadline=None):
    """Local search via random sign flips with acceptance criterion."""
    best = x.copy()
    best_score = _score(best)
    n = len(x)
    rng = np.random.RandomState(42)
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try flipping a random position
        idx = rng.randint(n)
        trial = best.copy()
        trial[idx] *= -1.0
        trial_score = _score(trial)
        
        if trial_score < best_score:
            best = trial
            best_score = trial_score
    
    return best, best_score

def _gradient_step(x, lr, deadline=None):
    """One projected gradient descent step using FFT gradient."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    
    F = np.fft.rfft(x, size)
    # Gradient of max|ac| is tricky; use surrogate: sum of |ac|^2 gradient
    # d/dx sum|ac|^2 = 2 * real(ifft(F * F.conj() * F.conj())) ... 
    # Actually for autoconv ac[k] = sum x[i]x[i+k], 
    # gradient of sum_k ac[k]^2 w.r.t. x[j] = 2 * sum_k ac[k] * (x[k-j] + x[k+j])
    # This is complex. Use a simpler approach: gradient of ||F||^4 / (sum x)^2
    
    # Simpler: use the fact that for fixed sum, we want to minimize max|ac|
    # Use subgradient of max|ac|: the ac value at the argmax position
    ac = _autoconv(x)
    abs_ac = np.abs(ac)
    max_idx = np.argmax(abs_ac)
    
    # Subgradient of |ac[max_idx]| w.r.t. x
    # ac[max_idx] = sum_i x[i] * x[i + max_idx] (with appropriate bounds)
    k = max_idx
    grad = np.zeros(n)
    for i in range(n):
        j = i + k
        if j < n:
            grad[i] += x[j]
            grad[j] += x[i]
        # Also for negative lag (but autoconv is symmetric, so this covers it)
    
    # Normalize
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-10:
        grad = grad / grad_norm
    
    x_new = x - lr * grad
    # Project to ±1
    x_new = np.sign(x_new)
    x_new[x_new == 0] = 1.0
    
    return x_new

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin out of 120s
    
    best_x = None
    best_score = 1e10
    
    # Multi-start with diverse random initializations
    n_starts = 20
    
    for start in range(n_starts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if start == 0:
            # Random ±1
            x = _random_signed(N, seed=start)
        elif start == 1:
            # Alternating pattern with random perturbation
            x = np.ones(N)
            x[::2] = -1.0
            # Randomly flip some
            rng = np.random.RandomState(start + 100)
            flip_mask = rng.rand(N) < 0.1
            x[flip_mask] *= -1.0
        elif start == 2:
            # Sine-based initialization
            t = np.linspace(0, 4*np.pi, N)
            x = np.sign(np.sin(t) + 0.3 * np.random.RandomState(start).randn(N))
        elif start == 3:
            # Block-based: alternating blocks of different lengths
            x = np.ones(N)
            block_sizes = [1, 2, 3, 1, 2, 4, 1, 3, 2, 1]
            pos = 0
            sign = 1.0
            for bs in block_sizes:
                end = min(pos + bs, N)
                x[pos:end] = sign
                sign *= -1.0
                pos = end
                if pos >= N:
                    break
            # Fill remaining randomly
            if pos < N:
                rng = np.random.RandomState(start + 200)
                x[pos:] = rng.choice([-1.0, 1.0], size=N - pos)
        else:
            # Pure random with different seeds
            x = _random_signed(N, seed=start + 50)
        
        # Evaluate initial
        init_score = _score(x)
        
        # Local search refinement
        x_ls, ls_score = _local_search(x, max_iter=5000, deadline=deadline)
        
        if ls_score < best_score:
            best_score = ls_score
            best_x = x_ls.copy()
        
        # If we have time, do gradient polishing on the best
        if time.time() < deadline - 10 and best_x is not None:
            x_grad = best_x.copy()
            lr = 0.1
            for giter in range(100):
                if time.time() > deadline:
                    break
                x_grad = _gradient_step(x_grad, lr, deadline=deadline)
                g_score = _score(x_grad)
                if g_score < best_score:
                    best_score = g_score
                    best_x = x_grad.copy()
                lr *= 0.95
    
    # Final local search polish on the best
    if best_x is not None and time.time() < deadline - 5:
        best_x, best_score = _local_search(best_x, max_iter=5000, deadline=deadline)
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
