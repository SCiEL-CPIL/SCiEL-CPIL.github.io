import numpy as np
import time

# Precompute bin indices for fast entropy calculation
_N = 10
_y_idx, _x_idx = np.indices((_N, _N))
_x_flat = _x_idx.flatten()
_y_flat = _y_idx.flatten()

# x - y ranges from -9 to 9, shift by 9 to get 0..18
_diff_shifted = _x_flat - _y_flat + 9  # 0..18
# x ranges from 0 to 9
_h0_vals = _x_flat  # 0..9
# x + y ranges from 0 to 18
_h1_vals = _x_flat + _y_flat  # 0..18
# x + 2y ranges from 0 to 27
_h2_vals = _x_flat + 2 * _y_flat  # 0..27

_N_BINS_DIFF = 19
_N_BINS_H0 = 10
_N_BINS_H1 = 19
_N_BINS_H2 = 28

def _entropy_from_bincounts(bin_counts, total_weight):
    """Compute Shannon entropy from bin counts."""
    if total_weight <= 1e-15:
        return 0.0
    probs = bin_counts / total_weight
    mask = probs > 1e-15
    if not np.any(mask):
        return 0.0
    p = probs[mask]
    return -np.sum(p * np.log2(p))

def _compute_ratio_fast(w):
    """Fast ratio computation using precomputed bin indices."""
    total = w.sum()
    if total <= 1e-15:
        return 0.0
    
    # Compute bin counts using bincount with precomputed indices
    diff_counts = np.bincount(_diff_shifted, weights=w, minlength=_N_BINS_DIFF)
    h0_counts = np.bincount(_h0_vals, weights=w, minlength=_N_BINS_H0)
    h1_counts = np.bincount(_h1_vals, weights=w, minlength=_N_BINS_H1)
    h2_counts = np.bincount(_h2_vals, weights=w, minlength=_N_BINS_H2)
    
    D = _entropy_from_bincounts(diff_counts, total)
    H0 = _entropy_from_bincounts(h0_counts, total)
    H1 = _entropy_from_bincounts(h1_counts, total)
    H2 = _entropy_from_bincounts(h2_counts, total)
    
    M = max(H0, H1, H2)
    if M <= 1e-9:
        return 0.0
    return D / M

def _compute_ratio(P):
    """Compute the ratio D/M for a given 10x10 weight matrix P."""
    P = np.clip(P, 0, None)
    s = P.sum()
    if s <= 1e-15:
        return 0.0
    P = P / s
    return _compute_ratio_fast(P.flatten())

def arithmetic_kakeya_distribution() -> np.ndarray:
    n = 10
    start_time = time.time()
    timeout = 600
    safety_margin = 10
    deadline = start_time + timeout - safety_margin
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Uniform
    candidates.append(np.ones((n, n), dtype=np.float64) / 100.0)
    
    # 2. Concentrate on x+2y = constant (minimizes H2)
    # Try several constants
    for c in range(0, 28, 2):
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i + 2*j == c:
                    w[i, j] = 1.0
        if w.sum() > 0:
            candidates.append(w / w.sum())
            
    # 3. Concentrate on x = constant (minimizes H0)
    for c in range(n):
        w = np.zeros((n, n), dtype=np.float64)
        w[:, c] = 1.0
        candidates.append(w / w.sum())
        
    # 4. Concentrate on x+y = constant (minimizes H1)
    for c in range(0, 18, 2):
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i + j == c:
                    w[i, j] = 1.0
        if w.sum() > 0:
            candidates.append(w / w.sum())
            
    # 5. "Thick" diagonals for x+2y
    for c in [4, 6, 8, 10, 12, 14]:
        w = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if abs(i + 2*j - c) <= 1:
                    w[i, j] = 1.0
        if w.sum() > 0:
            candidates.append(w / w.sum())
            
    # 6. Random sparse matrices
    np.random.seed(42)
    for _ in range(10):
        w = np.random.rand(n, n)
        w[w < 0.5] = 0
        if w.sum() > 0:
            candidates.append(w / w.sum())
            
    # Evaluate all candidates
    best_score = -1
    best_P = candidates[0]
    for c in candidates:
        sc = _compute_ratio_fast(c.flatten())
        if sc > best_score:
            best_score = sc
            best_P = c.copy()
    
    # Local search: Coordinate Ascent / Hill Climbing
    current_P = best_P.copy()
    current_score = best_score
    
    # We will perform iterative improvements.
    # Since the function is smooth enough, small steps work.
    # We use a mix of random coordinate updates and small global perturbations.
    
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 500
    patience = 0
    
    while time.time() < deadline:
        if no_improve_count > max_no_improve:
            # Restart or adjust strategy
            if current_score < best_score * 0.95:
                # Revert to best and add noise
                current_P = best_P + np.random.normal(0, 0.1, (n, n))
                current_P = np.clip(current_P, 0, None)
                s = current_P.sum()
                if s > 1e-9:
                    current_P /= s
                else:
                    current_P = best_P.copy()
                current_score = _compute_ratio_fast(current_P.flatten())
            else:
                # Try a larger jump
                current_P = best_P + np.random.normal(0, 0.5, (n, n))
                current_P = np.clip(current_P, 0, None)
                s = current_P.sum()
                if s > 1e-9:
                    current_P /= s
                else:
                    current_P = best_P.copy()
                current_score = _compute_ratio_fast(current_P.flatten())
            
            no_improve_count = 0
            max_no_improve = int(max_no_improve * 0.8)
            if max_no_improve < 100:
                break
            continue
        
        # Randomly select a coordinate to update
        i, j = np.random.randint(0, n, 2)
        
        # Try a few step sizes
        improved = False
        for delta in [step_size, -step_size, step_size*0.5, -step_size*0.5, step_size*2, -step_size*2]:
            new_P = current_P.copy()
            new_P[i, j] += delta
            new_P = np.clip(new_P, 0, None)
            s = new_P.sum()
            if s <= 1e-9:
                continue
            new_P = new_P / s
            
            new_score = _compute_ratio_fast(new_P.flatten())
            
            if new_score > current_score:
                current_P = new_P
                current_score = new_score
                no_improve_count = 0
                improved = True
                break
        
        if not improved:
            no_improve_count += 1
            # Decay step size slightly
            step_size = max(0.001, step_size * 0.98)
            
        if current_score > best_score:
            best_score = current_score
            best_P = current_P.copy()
            
    # Final validation and normalization
    result = np.clip(best_P, 0, None)
    s = result.sum()
    if s <= 1e-9:
        result = np.ones((n, n)) / 100.0
    else:
        result = result / s
    
    return result
