"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv_full(h):
    """Compute full linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return ac[:size]

def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution."""
    ac = _autoconv_full(h)
    return np.max(np.abs(ac))

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e15
    return _autoconv_max_abs(h) / (s * s)

def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p."""
    if n % p == 0:
        return 0
    return 1 if pow(n, (p - 1) // 2, p) == 1 else -1

def _legendre_sequence(N, p=None):
    """Generate a Legendre sequence of length N."""
    if p is None:
        p = N
        while True:
            if all(p % i != 0 for i in range(2, int(p**0.5) + 1)) and p > 1:
                break
            p += 1
    h = np.zeros(N)
    for i in range(N):
        h[i] = float(_legendre_symbol(i + 1, p))
    if abs(np.sum(h)) < N * 0.01:
        h = -h
    return h

def _m_sequence(N, degree=None):
    """Generate an M-sequence (maximal length sequence)."""
    if degree is None:
        degree = 1
        while (2**degree - 1) < N:
            degree += 1
    if degree == 8:
        taps = [8, 4, 3, 2, 1]
    elif degree == 7:
        taps = [7, 6, 1]
    elif degree == 6:
        taps = [6, 5, 1]
    elif degree == 5:
        taps = [5, 3, 1]
    else:
        return np.random.default_rng(42).choice([-1.0, 1.0], size=N)
    
    state = [1] * degree
    h = np.zeros(N)
    for i in range(N):
        h[i] = 1.0 if state[0] == 1 else -1.0
        new_bit = 0
        for t in taps:
            new_bit ^= state[t - 1]
        state = state[1:] + [new_bit]
    return h

def _local_search_fast(h, deadline, max_iters=500):
    """Fast local search with sign flips and zero toggles."""
    h = h.copy()
    best_val = _score(h)
    best_h = h.copy()
    
    rng = np.random.default_rng(54321)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        move_type = rng.random()
        idx = rng.integers(0, N)
        
        if move_type < 0.5:
            orig = h[idx]
            if orig == 0:
                continue
            h[idx] = -orig
            new_val = _score(h)
            if new_val < best_val - 1e-10:
                best_val = new_val
                best_h = h.copy()
            else:
                h[idx] = orig
        
        elif move_type < 0.8:
            orig = h[idx]
            if orig == 0:
                h[idx] = rng.choice([-1.0, 1.0])
            else:
                h[idx] = 0.0
            new_val = _score(h)
            if new_val < best_val - 1e-10:
                best_val = new_val
                best_h = h.copy()
            else:
                h[idx] = orig
        
        else:
            orig = h[idx]
            candidates = [0.0, 0.5, -0.5, 1.0, -1.0]
            found_improvement = False
            for c in candidates:
                if abs(c - orig) < 1e-10:
                    continue
                h[idx] = c
                new_val = _score(h)
                if new_val < best_val - 1e-10:
                    best_val = new_val
                    best_h = h.copy()
                    found_improvement = True
                    break
            if not found_improvement:
                h[idx] = orig
    
    return best_h, best_val

def _local_search_polish(h, deadline, max_iters=1000):
    """High-precision polish pass."""
    h = h.copy()
    best_val = _score(h)
    best_h = h.copy()
    
    rng = np.random.default_rng(99999)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        idx = rng.integers(0, N)
        orig = h[idx]
        
        if orig != 0:
            h[idx] = -orig
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                best_h = h.copy()
            else:
                h[idx] = orig
        
        if orig == 0:
            h[idx] = rng.choice([-1.0, 1.0])
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                best_h = h.copy()
            else:
                h[idx] = 0.0
        else:
            h[idx] = 0.0
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
                best_h = h.copy()
            else:
                h[idx] = orig
    
    return best_h, best_val

def _smooth_local_search(h, deadline, max_iters=300):
    """Smooth local search using gradient-like moves on continuous values."""
    h = h.copy()
    best_val = _score(h)
    best_h = h.copy()
    rng = np.random.default_rng(77777)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Pick a random index and try to adjust its value
        idx = rng.integers(0, N)
        orig = h[idx]
        
        # Try several nearby values
        for step in [0.1, -0.1, 0.2, -0.2, 0.3, -0.3]:
            new_val_candidate = orig + step
            if new_val_candidate < -2.0 or new_val_candidate > 2.0:
                continue
            h[idx] = new_val_candidate
            new_score = _score(h)
            if new_score < best_val - 1e-10:
                best_val = new_score
                best_h = h.copy()
                orig = new_val_candidate
                break
            h[idx] = orig
    
    return best_h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Generate diverse initializations
    initializations = []
    
    for p in [401, 409, 421, 433, 449]:
        h = _legendre_sequence(N, p)
        initializations.append(h)
    
    for deg in [7, 8]:
        h = _m_sequence(N, deg)
        initializations.append(h)
    
    for _ in range(10):
        h = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(h)
        if abs(s) < N * 0.05:
            flip_idx = rng.choice(N, size=N//10, replace=False)
            h[flip_idx] = 1.0
        initializations.append(h)
    
    for density in [0.2, 0.3, 0.4]:
        for _ in range(5):
            mask = rng.random(N) < density
            h = np.zeros(N)
            h[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            if abs(np.sum(h)) < 1e-6:
                continue
            initializations.append(h)
    
    for _ in range(8):
        h = np.ones(N)
        pos = 0
        while pos < N:
            block_len = rng.integers(2, N // 6 + 1)
            sign = rng.choice([-1.0, 1.0])
            h[pos:pos + block_len] = sign
            pos += block_len
        initializations.append(h)
    
    for freq in [0.3, 0.7, 1.3, 2.1]:
        x = np.arange(N)
        h = np.sign(np.cos(2 * np.pi * freq * x / N))
        if abs(np.sum(h)) < N * 0.02:
            h = np.sign(h + 0.1)
        initializations.append(h)
    
    # Evaluate all initializations
    candidates = []
    for h in initializations:
        val = _score(h)
        candidates.append((val, h))
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:15]
    
    # Phase 2: Local search on top candidates
    for val_init, h_init in top_candidates:
        if time.time() > deadline - 15:
            break
        
        remaining = deadline - 10 - time.time()
        if remaining < 2:
            break
        max_iters = max(100, int(remaining / len(top_candidates) * 200))
        h_opt, val_opt = _local_search_fast(h_init, deadline - 10, max_iters=max_iters)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 3: Smooth local search for continuous refinement
    if time.time() < deadline - 15:
        h_smooth, val_smooth = _smooth_local_search(best_h, deadline - 10, max_iters=200)
        if val_smooth < best_val:
            best_val = val_smooth
            best_h = h_smooth.copy()
    
    # Phase 4: Final polish
    if time.time() < deadline - 5:
        h_final, val_final = _local_search_polish(best_h, deadline - 2, max_iters=1500)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h
# EVOLVE-BLOCK-END
