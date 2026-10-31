"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return float(np.max(np.abs(ac)))


def _score(h):
    """Compute max|autoconv| / (sum h)^2. Lower is better."""
    s = float(np.sum(h))
    if abs(s) < 1e-9:
        return 1e15
    return _autoconv_max_abs(h) / (s * s)


def _gen_binary_random(n, seed):
    """Generate a random binary sequence with values in {-1, 1}."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n)


def _gen_sparse_random(n, density, seed):
    """Generate a sparse sequence with given density and random signs."""
    rng = np.random.default_rng(seed)
    mask = rng.random(n) < density
    x = np.zeros(n)
    x[mask] = rng.choice([-1.0, 1.0], size=int(np.sum(mask)))
    return x


def _gen_legendre(n):
    """Generate Legendre symbol sequence of length n."""
    p = n
    while True:
        # Check if p is prime
        if p < 2:
            p += 1
            continue
        is_p = True
        for i in range(2, int(np.sqrt(p)) + 1):
            if p % i == 0:
                is_p = False
                break
        if is_p:
            break
        p += 1
    
    x = np.zeros(n)
    x[0] = 1.0
    for i in range(1, n):
        # Legendre symbol (i|p)
        leg = pow(i, (p - 1) // 2, p)
        x[i] = 1.0 if leg == 1 else -1.0
    return x


def _gen_chirp(n, freq_start=0.1, freq_end=0.5):
    """Generate a chirp-like binary sequence."""
    t = np.linspace(0, 1, n)
    # Phase that accelerates
    phase = 2 * np.pi * (freq_start * t + (freq_end - freq_start) * t**2 / 2)
    return np.sign(np.sin(phase))


def _gen_barker_tile(n):
    """Tile a Barker sequence of length 13."""
    bark = np.array([1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=float)
    x = np.tile(bark, (n // len(bark) + 1))[:n]
    return x


def _local_search_flip(h, max_passes=50, deadline=None, patience=5):
    """
    Local search: try flipping each element. Keep flips that improve score.
    Uses incremental FFT update for speed.
    """
    h = h.copy()
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    
    best_val = _score(h)
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    current_max = float(np.max(np.abs(ac)))
    s = float(np.sum(h))
    
    no_improve_count = 0
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            
            orig = h[i]
            h[i] = -orig
            delta = -2 * orig
            
            # Update FFT incrementally
            H_new = H + delta * np.fft.rfft(np.eye(n)[i], fsize)
            # Actually, let's just recompute for simplicity but check deadline more often
            # For speed, let's use a faster check
            
            # Recompute score
            new_val = _score(h)
            
            if new_val < best_val - 1e-12:
                best_val = new_val
                H = H_new
                improved = True
                no_improve_count = 0
            else:
                h[i] = orig
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= patience:
                break
    
    return h, best_val


def _local_search_sparse(h, max_passes=30, deadline=None):
    """
    Local search allowing 0, ±1 values.
    Try: flip sign, set to 0, set from 0 to ±1.
    """
    h = h.copy()
    best_val = _score(h)
    
    for p in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try flipping non-zero elements
        for i in range(len(h)):
            if deadline is not None and time.time() > deadline:
                break
            if h[i] != 0:
                orig = h[i]
                h[i] = -orig
                new_val = _score(h)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = orig
        
        # Try setting non-zero to zero
        for i in range(len(h)):
            if deadline is not None and time.time() > deadline:
                break
            if h[i] != 0:
                orig = h[i]
                h[i] = 0.0
                new_val = _score(h)
                if new_val < best_val - 1e-12:
                    best_val = new_val
                    improved = True
                else:
                    h[i] = orig
        
        # Try setting zero to ±1
        for i in range(len(h)):
            if deadline is not None and time.time() > deadline:
                break
            if h[i] == 0:
                for val in [1.0, -1.0]:
                    h[i] = val
                    new_val = _score(h)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        improved = True
                        break
                    else:
                        h[i] = 0.0
        
        if not improved:
            break
    
    return h, best_val


def _simulated_annealing(h, deadline, start_temp=1.0, cool_rate=0.95, min_temp=0.01):
    """
    Simulated annealing for binary sequences.
    """
    h = h.copy()
    best_h = h.copy()
    best_val = _score(h)
    current_val = best_val
    temp = start_temp
    
    n = len(h)
    
    while temp > min_temp:
        if time.time() > deadline:
            break
        
        # Random move: flip a random element
        i = np.random.randint(n)
        h[i] = -h[i]
        new_val = _score(h)
        
        delta = new_val - current_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / temp):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_h = h.copy()
        else:
            h[i] = -h[i]
        
        temp *= cool_rate
    
    return best_h, best_val


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    best_x = None
    best_val = 1e15

    # Phase 1: Generate diverse candidates
    candidates = []
    
    # Algebraic constructions
    try:
        candidates.append(_gen_legendre(N))
    except:
        pass
    
    candidates.append(_gen_barker_tile(N))
    candidates.append(_gen_chirp(N))
    
    # Random binary sequences
    for seed in range(100):
        candidates.append(_gen_binary_random(N, seed))
    
    # Sparse sequences
    for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for seed in range(20):
            x = _gen_sparse_random(N, density, seed)
            if abs(np.sum(x)) > 1e-6:
                candidates.append(x)
    
    # All ones
    candidates.append(np.ones(N))
    
    # Evaluate candidates
    for x in candidates:
        if time.time() > deadline - 30:
            break
        val = _score(x)
        if val < best_val:
            best_val = val
            best_x = x.copy()
    
    # Phase 2: Local search on best candidate
    if time.time() < deadline - 10:
        x_ls, val_ls = _local_search_flip(best_x.copy(), max_passes=20, deadline=deadline - 10)
        if val_ls < best_val:
            best_val = val_ls
            best_x = x_ls
    
    # Phase 3: Sparse local search
    if time.time() < deadline - 5:
        x_sp, val_sp = _local_search_sparse(best_x.copy(), max_passes=10, deadline=deadline - 5)
        if val_sp < best_val:
            best_val = val_sp
            best_x = x_sp
    
    # Phase 4: Simulated annealing from multiple starting points
    if time.time() < deadline - 10:
        # Try SA from best and a few random starts
        for start_idx in range(3):
            if time.time() > deadline - 5:
                break
            
            if start_idx == 0:
                h_init = best_x.copy()
            else:
                h_init = _gen_binary_random(N, 1000 + start_idx)
            
            x_sa, val_sa = _simulated_annealing(h_init, deadline - 5)
            if val_sa < best_val:
                best_val = val_sa
                best_x = x_sa
    
    # Phase 5: Final polish
    if time.time() < deadline - 2:
        x_final, val_final = _local_search_flip(best_x.copy(), max_passes=10, deadline=deadline - 2)
        if val_final < best_val:
            best_val = val_final
            best_x = x_final
    
    # Ensure valid result
    if best_x is None or abs(np.sum(best_x)) < 1e-6:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
