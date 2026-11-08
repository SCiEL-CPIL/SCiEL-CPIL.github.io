"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin
    
    # Compute the score for a given height vector
    def score(h):
        h = np.asarray(h, dtype=np.float64)
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-15:
            return 0.0
        # Autoconvolution (discrete)
        # (h*h)_k = sum_i h[i] * h[i-k] for valid indices
        # Using np.convolve with mode='full'
        conv = np.convolve(h, h, mode='full')
        sup_norm = np.max(np.abs(conv))
        if sup_norm < 1e-15:
            return 0.0
        return l2_sq / (l1 * sup_norm)
    
    # Try several candidate strategies
    best_h = np.ones(N)
    best_score = score(best_h)
    
    # Strategy 1: Single spike
    for i in range(N):
        h = np.zeros(N)
        h[i] = 1.0
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Strategy 2: Two spikes
    for i in range(N):
        for j in range(i+1, N):
            h = np.zeros(N)
            h[i] = 1.0
            h[j] = 1.0
            s = score(h)
            if s > best_score:
                best_score = s
                best_h = h.copy()
    
    # Strategy 3: Random search with local refinement
    np.random.seed(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        # Random initial point
        h = np.random.exponential(1.0, N)
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Strategy 4: Coordinate descent / local optimization from best
    h = best_h.copy()
    for it in range(500):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_val = h[i]
            # Try a few perturbations
            for delta in [-0.5, -0.1, 0.1, 0.5, 1.0]:
                new_val = max(0.0, old_val + delta)
                h[i] = new_val
                s = score(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val
        if not improved:
            break
    
    # Strategy 5: Try patterns - alternating, triangular, etc.
    patterns = [
        np.ones(N),
        np.zeros(N),  # will be replaced
        np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)]),
        np.array([1.0 - abs(i - N//2)/(N//2) for i in range(N)]),
        np.array([1.0/np.sqrt(i+1) for i in range(N)]),
        np.array([np.exp(-i/10) for i in range(N)]),
    ]
    for p in patterns:
        if np.sum(p) < 1e-15:
            continue
        s = score(p)
        if s > best_score:
            best_score = s
            best_h = p.copy()
    
    # Strategy 6: More extensive random search
    np.random.seed(123)
    for trial in range(500):
        if time.time() > deadline:
            break
        # Mix of sparse and dense
        if np.random.random() < 0.5:
            # Sparse: few non-zero entries
            k = np.random.randint(1, 6)
            idx = np.random.choice(N, k, replace=False)
            h = np.zeros(N)
            h[idx] = np.random.exponential(1.0, k)
        else:
            # Dense with random structure
            h = np.random.exponential(1.0, N)
            # Maybe zero out some
            mask = np.random.random(N) < 0.5
            h[mask] = 0.0
        s = score(h)
        if s > best_score:
            best_score = s
            best_h = h.copy()
    
    # Strategy 7: Local refinement with smaller steps
    h = best_h.copy()
    for it in range(1000):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            old_val = h[i]
            for delta in [-0.01, -0.001, 0.001, 0.01]:
                new_val = max(0.0, old_val + delta)
                h[i] = new_val
                s = score(h)
                if s > best_score:
                    best_score = s
                    best_h = h.copy()
                    improved = True
                else:
                    h[i] = old_val
        if not improved:
            break
    
    # Normalize doesn't change the ratio, so return as-is
    return list(best_h)
# EVOLVE-BLOCK-END
