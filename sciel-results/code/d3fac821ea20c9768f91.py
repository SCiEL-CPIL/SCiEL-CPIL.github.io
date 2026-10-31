"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.signal import fftconvolve

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

def _soft_max_abs(ac, beta=10.0):
    """Soft approximation of max|ac| using log-sum-exp."""
    x = np.abs(ac)
    m = np.max(x)
    if m < 1e-15:
        return 0.0
    # soft max: m + (1/beta) * log(sum(exp(beta*(x-m))))
    shifted = beta * (x - m)
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    s = np.sum(np.exp(shifted))
    return m + (1.0 / beta) * np.log(s)

def _local_search(h, deadline, max_iters=200, beta=10.0):
    """Multi-move local search: sign flip, zero toggle, magnitude adjust."""
    h = h.copy()
    best_val = _score(h)
    
    rng = np.random.default_rng(12345)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        move_type = rng.random()
        idx = rng.integers(0, N)
        
        if move_type < 0.5:
            # Sign flip
            orig = h[idx]
            h[idx] = -orig
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
            else:
                h[idx] = orig
        
        elif move_type < 0.75:
            # Toggle between zero and +/-1
            if h[idx] == 0:
                h[idx] = rng.choice([-1.0, 1.0])
            else:
                h[idx] = 0.0
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
            else:
                # Revert
                if move_type < 0.625:
                    h[idx] = -h[idx] if h[idx] != 0 else 1.0  # This won't work cleanly
                # Actually let's just track state properly
                pass
        
        else:
            # Magnitude adjustment (for non-zero entries)
            if h[idx] != 0:
                orig = h[idx]
                # Try scaling toward 0 or away
                h[idx] = orig * 0.5
                new_val = _score(h)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                else:
                    h[idx] = orig * 1.5
                    new_val = _score(h)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                    else:
                        h[idx] = orig
    
    return h, best_val

def _local_search_v2(h, deadline, max_iters=300):
    """Improved local search with proper state tracking."""
    h = h.copy()
    best_val = _score(h)
    
    rng = np.random.default_rng(54321)
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        move_type = rng.random()
        idx = rng.integers(0, N)
        
        if move_type < 0.4:
            # Sign flip
            orig = h[idx]
            h[idx] = -orig
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
            else:
                h[idx] = orig
        
        elif move_type < 0.7:
            # Toggle zero/non-zero
            orig = h[idx]
            if orig == 0:
                h[idx] = rng.choice([-1.0, 1.0])
            else:
                h[idx] = 0.0
            new_val = _score(h)
            if new_val < best_val - 1e-12:
                best_val = new_val
            else:
                h[idx] = orig
        
        else:
            # Try setting to a specific value
            orig = h[idx]
            candidates = [0.0, 1.0, -1.0, 0.5, -0.5]
            for c in candidates:
                if c == orig:
                    continue
                h[idx] = c
                new_val = _score(h)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    break
            else:
                h[idx] = orig
    
    return h, best_val

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin
    
    best_h = np.ones(N)
    best_val = _score(best_h)
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Try diverse initializations
    initializations = []
    
    # Dense random ±1
    for _ in range(20):
        h = rng.choice([-1.0, 1.0], size=N)
        s = np.sum(h)
        if abs(s) < N * 0.05:
            h = np.sign(h + 0.3)
        initializations.append(h)
    
    # Sparse sequences with various densities
    for density in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for _ in range(15):
            mask = rng.random(N) < density
            h = np.zeros(N)
            h[mask] = rng.choice([-1.0, 1.0], size=np.sum(mask))
            s = np.sum(h)
            if abs(s) < 1e-6:
                continue
            initializations.append(h)
    
    # Block patterns
    for _ in range(15):
        h = np.ones(N)
        pos = 0
        while pos < N:
            block_len = rng.integers(1, N // 8 + 1)
            sign = rng.choice([-1.0, 1.0])
            h[pos:pos + block_len] = sign
            pos += block_len
        initializations.append(h)
    
    # Chirp-like patterns
    for freq in [0.5, 1.0, 2.0, 3.0]:
        x = np.arange(N)
        h = np.sign(np.cos(2 * np.pi * freq * x / N))
        initializations.append(h)
    
    # Evaluate all initializations and keep top candidates
    candidates = []
    for h in initializations:
        val = _score(h)
        candidates.append((val, h))
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Sort by score and take top 30 for local search
    candidates.sort(key=lambda x: x[0])
    top_candidates = candidates[:30]
    
    # Phase 2: Local search on top candidates
    for val_init, h_init in top_candidates:
        if time.time() > deadline - 5:
            break
        
        h_opt, val_opt = _local_search_v2(h_init, deadline - 2, max_iters=200)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
    
    # Phase 3: Final polish on the best solution
    if time.time() < deadline - 3:
        h_final, val_final = _local_search_v2(best_h, deadline - 1, max_iters=500)
        if val_final < best_val:
            best_val = val_final
            best_h = h_final
    
    return best_h
# EVOLVE-BLOCK-END
