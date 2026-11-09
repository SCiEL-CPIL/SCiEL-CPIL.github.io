import numpy as np

def _run_bfd(demands, capacity):
    """Run Best-Fit-Decreasing and return item-to-bin mapping."""
    n = len(demands)
    order = np.argsort(-demands, kind="stable")
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    
    for idx in order:
        size = demands[idx]
        best_bi = -1
        best_rem = np.inf
        for bi in range(len(bin_loads)):
            rem = capacity - bin_loads[bi]
            if rem >= size and rem < best_rem:
                best_rem = rem
                best_bi = bi
        if best_bi >= 0:
            bin_loads[best_bi] += size
            item_bin[idx] = best_bi
        else:
            bin_loads.append(size)
            item_bin[idx] = len(bin_loads) - 1
    return item_bin

def _run_ffd(demands, capacity):
    """Run First-Fit-Decreasing and return item-to-bin mapping."""
    n = len(demands)
    order = np.argsort(-demands, kind="stable")
    bin_loads = []
    item_bin = np.full(n, -1, dtype=int)
    
    for idx in order:
        size = demands[idx]
        for bi in range(len(bin_loads)):
            if capacity - bin_loads[bi] >= size:
                bin_loads[bi] += size
                item_bin[idx] = bi
                break
        else:
            bin_loads.append(size)
            item_bin[idx] = len(bin_loads) - 1
    return item_bin

def heuristics(demands, capacity):
    d = np.asarray(demands, dtype=float)
    n = len(d)
    
    # 1. Compute reference packings
    item_bin_bfd = _run_bfd(d, capacity)
    item_bin_ffd = _run_ffd(d, capacity)
    
    # 2. Vectorized Heuristic Calculation
    # d_i shape (N, 1), d_j shape (1, N)
    d_i = d[:, np.newaxis]
    d_j = d[np.newaxis, :]
    
    sums = d_i + d_j
    valid = sums <= capacity
    residuals = capacity - sums
    residuals[~valid] = np.inf
    
    min_d = np.min(d)
    
    # Calculate Fillability Score
    # We want to reward residuals that are 0, or >= min_d and close to an existing item size.
    # Flatten residuals for distance calculation
    res_flat = residuals.flatten()
    
    # Distance from each residual to each item size
    # res_flat shape (N*N,), d shape (N,)
    # Distances shape (N*N, N)
    distances = np.abs(res_flat[:, np.newaxis] - d[np.newaxis, :])
    min_dist = np.min(distances, axis=1)
    min_dist = min_dist.reshape(n, n)
    
    # Base fill score: closer to an item size is better
    fill_score = 1.0 / (1.0 + min_dist)
    
    # Special cases
    is_zero = residuals < 1e-6
    is_dead = (residuals > 1e-6) & (residuals < min_d)
    
    # Perfect fit gets a high score
    fill_score[is_zero] = 5.0
    # Dead space (too small for any item) gets a low score
    fill_score[is_dead] = 0.2
    
    # Calculate Tightness Score
    # Smaller residual is generally better, but fill_score handles the "quality"
    # Tightness: 1 / (1 + residual/capacity)
    tight_score = 1.0 / (1.0 + residuals / capacity)
    tight_score[~valid] = 0.0
    
    # Combine
    base_score = tight_score * fill_score
    
    # 3. Consensus Boost
    # Create matrices of bin assignments
    bin_i_bfd = item_bin_bfd[:, np.newaxis]
    bin_j_bfd = item_bin_bfd[np.newaxis, :]
    same_bfd = (bin_i_bfd == bin_j_bfd)
    
    bin_i_ffd = item_bin_ffd[:, np.newaxis]
    bin_j_ffd = item_bin_ffd[np.newaxis, :]
    same_ffd = (bin_i_ffd == bin_j_ffd)
    
    # Boost factors
    # 4.0 if both, 2.0 if one, 0.7 if none
    boost = np.where(same_bfd & same_ffd, 4.0, 
                     np.where(same_bfd | same_ffd, 2.0, 0.7))
                     
    H = base_score * boost
    
    # 4. Cleanup
    H[~valid] = 1e-9
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
