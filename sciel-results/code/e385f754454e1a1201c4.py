import numpy as np
import time

N = 50
M = 2 * N - 1  # length of g = 99

# EVOLVE-BLOCK-START
def _compute_value_fast(h):
    """Compute objective value for step heights h (length N)."""
    h = np.asarray(h, dtype=np.float64)
    if not np.all(np.isfinite(h)) or np.any(h < -1e-15):
        return -1.0
    h = np.maximum(h, 0.0)
    g = np.convolve(h, h)
    m = len(g)  # 99
    # l2 = (w/3) * (2*sum(g^2) + sum_{i=1}^{m-1} g_i g_{i+1})
    w = 1.0 / (m + 1)
    sum_g2 = np.dot(g, g)
    sum_cross = np.dot(g[:-1], g[1:])
    l2 = (w / 3.0) * (2.0 * sum_g2 + sum_cross)
    norm_1 = np.sum(g) / (m + 1)
    norm_inf = np.max(g)
    if norm_1 * norm_inf < 1e-30:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def _local_search(h, max_iter=5000, tol=1e-12):
    """Coordinate descent local search."""
    h = np.array(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    best_val = _compute_value_fast(h)
    if best_val <= 0:
        return h, best_val
    
    for it in range(max_iter):
        improved = False
        for i in range(N):
            # Try a few step sizes around current value
            base = h[i]
            # Perturbations: relative and absolute
            candidates = [
                base * 0.9, base * 1.1,
                base - 0.01, base + 0.01,
                base * 0.5, base * 2.0,
                0.0,
            ]
            # Also try golden-ratio-like steps
            if base > 0:
                candidates.extend([base * (1 + 0.01), base * (1 - 0.01)])
            
            best_cand = None
            best_cv = best_val
            for c in candidates:
                if c < -1e-15:
                    continue
                c = max(c, 0.0)
                if abs(c - base) < 1e-15:
                    continue
                h[i] = c
                cv = _compute_value_fast(h)
                if cv > best_cv + 1e-14:
                    best_cv = cv
                    best_cand = c
            h[i] = base
            if best_cand is not None:
                h[i] = best_cand
                best_val = best_cv
                improved = True
        
        if not improved:
            break
    
    return h, best_val

def _random_init(rng):
    """Generate a random initial point."""
    # Mix of different shapes
    x = np.linspace(-1, 1, N)
    kind = rng.integers(0, 6)
    if kind == 0:
        # Triangular
        h = np.maximum(0, 1 - np.abs(x))
    elif kind == 1:
        # Gaussian-like
        sigma = 0.3 + 0.4 * rng.random()
        h = np.exp(-x**2 / (2 * sigma**2))
    elif kind == 2:
        # Flat top
        width = 0.3 + 0.5 * rng.random()
        h = np.where(np.abs(x) < width, 1.0, 0.0)
        # Smooth edges
        h = np.convolve(h, np.ones(3)/3, mode='same')
    elif kind == 3:
        # Cosine hump
        h = np.maximum(0, np.cos(np.pi * x / 2))
    elif kind == 4:
        # Bump with noise
        h = np.maximum(0, 1 - np.abs(x)) + 0.1 * rng.random(N)
    else:
        # Pure random nonnegative
        h = rng.random(N)
    
    h = np.maximum(h, 0.0)
    if np.max(h) > 0:
        h = h / np.max(h)  # normalize peak to 1
    return h

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin
    
    rng = np.random.RandomState(42)
    
    best_h = None
    best_val = 0.0
    
    # Start with seed
    h_seed = np.ones(N)
    v_seed = _compute_value_fast(h_seed)
    if v_seed > best_val:
        best_val = v_seed
        best_h = h_seed.copy()
    
    # Local search from seed
    h_ls, v_ls = _local_search(h_seed, max_iter=3000)
    if v_ls > best_val:
        best_val = v_ls
        best_h = h_ls.copy()
    
    # Multiple random restarts
    n_restarts = 20
    for r in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        h_init = _random_init(rng)
        
        # Quick local search
        h_ls, v_ls = _local_search(h_init, max_iter=2000)
        if v_ls > best_val:
            best_val = v_ls
            best_h = h_ls.copy()
        
        # If promising, do more iterations
        if v_ls > best_val - 0.01:
            if time.time() > deadline - 5.0:
                break
            h_ls2, v_ls2 = _local_search(h_ls, max_iter=5000)
            if v_ls2 > best_val:
                best_val = v_ls2
                best_h = h_ls2.copy()
    
    # Final polish on best
    if time.time() < deadline - 2.0 and best_h is not None:
        h_final, v_final = _local_search(best_h, max_iter=3000)
        if v_final > best_val:
            best_val = v_final
            best_h = h_final.copy()
    
    if best_h is None:
        best_h = np.ones(N)
    
    return list(np.maximum(best_h, 0.0))
# EVOLVE-BLOCK-END
