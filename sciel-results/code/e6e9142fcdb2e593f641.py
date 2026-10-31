"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    H = np.fft.rfft(h, n=fft_size)
    ac = np.fft.irfft(H * H, n=fft_size)[:2*n-1]
    return np.max(np.abs(ac))

def _squared_integral(h):
    return np.sum(h) ** 2

def _score(h):
    """Compute max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-6:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)

def _random_binary_pattern(n_pos, rng):
    """Generate a random binary pattern with n_pos ones and rest -1."""
    h = np.full(N, -1.0)
    idx = rng.choice(N, size=n_pos, replace=False)
    h[idx] = 1.0
    return h

def _greedy_improve(h, max_iter=50, deadline=None):
    """Try flipping individual elements to improve the score."""
    best_h = h.copy()
    best_val = _score(best_h)
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Try flipping a random subset of positions
        n_tries = min(200, N)
        for _ in range(n_tries):
            j = np.random.randint(N)
            trial = best_h.copy()
            trial[j] = -trial[j]
            val = _score(trial)
            if val < best_val:
                best_h = trial
                best_val = val
                improved = True
        if not improved:
            break
    return best_h, best_val

def _local_search_continuous(h, max_iter=100, deadline=None):
    """Local search by perturbing individual values."""
    best_h = h.copy()
    best_val = _score(best_h)
    lr = 0.1
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for _ in range(50):
            j = np.random.randint(N)
            delta = np.random.uniform(-lr, lr)
            trial = best_h.copy()
            trial[j] += delta
            val = _score(trial)
            if val < best_val:
                best_h = trial
                best_val = val
                improved = True
        if not improved:
            lr *= 0.5
            if lr < 1e-4:
                break
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    rng = np.random.default_rng(42)
    
    best_h = np.ones(N)
    best_val = _score(best_h)  # Should be 1/N = 0.0025
    
    # Try various analytical patterns
    patterns = []
    
    # All ones (baseline)
    patterns.append(np.ones(N))
    
    # Ramp up then down (triangular)
    tri = np.concatenate([np.linspace(1, 0, N//2), np.linspace(0, 1, N - N//2)])
    patterns.append(tri)
    
    # Cosine taper
    x = np.arange(N)
    cos_taper = 0.5 * (1 + np.cos(2 * np.pi * x / N))
    patterns.append(cos_taper)
    
    # Half ones, half negative
    half_neg = np.ones(N)
    half_neg[N//2:] = -1
    patterns.append(half_neg)
    
    # Quarter positive, quarter negative, quarter positive, quarter negative
    q = np.ones(N)
    q[N//4:2*N//4] = -1
    q[3*N//4:] = -1
    patterns.append(q)
    
    # Gaussian-like
    center = N // 2
    gauss = np.exp(-0.5 * ((x - center) / (N/6))**2)
    patterns.append(gauss)
    
    # Sine wave
    patterns.append(np.sin(2 * np.pi * x / N))
    patterns.append(np.sin(2 * np.pi * x / (N/2)))
    
    # Two bumps
    two_bump = np.zeros(N)
    two_bump[:N//4] = 1
    two_bump[3*N//4:] = 1
    patterns.append(two_bump)
    
    # Random binary patterns with various densities
    for density in [0.3, 0.4, 0.5, 0.6, 0.7]:
        n_pos = int(density * N)
        h_rand = _random_binary_pattern(n_pos, rng)
        patterns.append(h_rand)
    
    # Evaluate all patterns
    for p in patterns:
        s = np.sum(p)
        if abs(s) < 1e-6:
            continue
        val = _score(p)
        if val < best_val:
            best_val = val
            best_h = p.copy()
    
    # Greedy improvement from best pattern
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        improved_h, improved_val = _greedy_improve(best_h, max_iter=30, deadline=deadline - 15)
        if improved_val < best_val:
            best_val = improved_val
            best_h = improved_h
    
    # Continuous local search
    elapsed = time.time() - start_time
    if elapsed < deadline - 15:
        cont_h, cont_val = _local_search_continuous(best_h, max_iter=50, deadline=deadline - 5)
        if cont_val < best_val:
            best_val = cont_val
            best_h = cont_h
    
    # More random restarts if time permits
    elapsed = time.time() - start_time
    while elapsed < deadline - 20:
        # Try random binary patterns
        for _ in range(5):
            density = rng.uniform(0.3, 0.7)
            n_pos = int(density * N)
            h_rand = _random_binary_pattern(n_pos, rng)
            val = _score(h_rand)
            if val < best_val:
                best_val = val
                best_h = h_rand.copy()
        
        # Greedy improve
        if time.time() < deadline - 10:
            improved_h, improved_val = _greedy_improve(best_h, max_iter=10, deadline=deadline - 5)
            if improved_val < best_val:
                best_val = improved_val
                best_h = improved_h
        
        elapsed = time.time() - start_time
    
    return best_h
# EVOLVE-BLOCK-END
