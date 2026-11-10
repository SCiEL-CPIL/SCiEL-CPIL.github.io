import numpy as np
import time


def _compute_area(positions, n=64, n_heights=20000):
    """Compute the union area using quadrature."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    
    union_lengths = w + np.sum(overlaps, axis=0)
    area = np.mean(union_lengths)
    
    return area


def _compute_area_fast(positions, n=64, n_heights=20000):
    """Faster area computation using precomputed constants."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)
    overlaps = np.minimum(diffs, w[np.newaxis, :])
    
    union_lengths = w + np.sum(overlaps, axis=0)
    area = np.mean(union_lengths)
    
    return area


def _subgradient(positions, n=64, n_heights=20000):
    """
    Compute subgradient of the union area with respect to positions.
    The area is piecewise linear in each x_i, so we can compute
    which triangles are in the 'active' set (affecting union boundary).
    """
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1 - y) / n
    
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort and track original indices
    sort_idx = np.argsort(L, axis=0)  # (n, n_heights)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    
    # For the union length: w + sum_j min(L_{j+1} - L_j, w)
    # The derivative w.r.t. x_i depends on which sorted positions i occupies
    # and whether the gap to neighbor is < w (overlapping) or > w (separated)
    
    # For each height, compute contribution of each triangle to the union boundary
    # A triangle's right edge contributes if it's the rightmost in its "cluster"
    # A triangle's left edge contributes if it's the leftmost in its "cluster"
    
    # Simpler approach: use the fact that the union of intervals has length
    # = sum of lengths - sum of overlaps. But with min formulation, let's
    # compute the subgradient numerically via finite differences for reliability,
    # but use a smarter scheme.
    
    # Actually, let's use the analytical structure:
    # Union length at height y = w + sum_{j=0}^{n-2} min(d_j, w) where d_j = L_{(j+1)} - L_{(j)}
    # d_j is linear in positions. min(d_j, w) has subgradient:
    #   1 w.r.t. L_{(j+1)}, -1 w.r.t. L_{(j)} if d_j < w
    #   0 if d_j > w
    #   [0,1] and [-1,0] if d_j == w
    
    # L_i = x_i + a_i * y, so dL_i/dx_i = 1
    
    # For each height, we need to know for each sorted position j,
    # which original triangle it is, and whether the gap to its neighbors is < w
    
    # Build subgradient array (n, n_heights)
    sg = np.zeros((n, n_heights))
    
    for m in range(n_heights):
        # Get sorted order at this height
        order = sort_idx[:, m]  # (n,)
        
        # For each consecutive pair in sorted order, check if gap < w
        for j in range(n - 1):
            d_j = diffs[j, m]
            if d_j < w[m]:  # overlapping: contributes to union
                # The gap contributes: +1 to the right triangle, -1 to the left
                left_tri = order[j]
                right_tri = order[j + 1]
                sg[right_tri, m] += 1.0
                sg[left_tri, m] -= 1.0
            # if d_j >= w, no contribution (separated intervals)
    
    # The base term w doesn't depend on positions
    # Area = mean over heights of union_lengths
    # d(area)/d(x_i) = mean over heights of sg[i, m]
    
    gradient = np.mean(sg, axis=1)
    
    return gradient


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Direct construction for Kakeya needle triangle positions.
    
    Uses the structural insight that triangles with larger a_i tilt more,
    so we offset their base positions to maximize overlap at all heights.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 590  # Safety margin before 600s limit
    
    a = (np.arange(n) + 1) / n
    
    # Try multiple initializations:
    # 1. Aligned: x_i = -a_i + 0.5 (near-optimal overlap by construction)
    # 2. Scaled: x_i = -a_i * c for various c
    # 3. Random perturbations of aligned
    
    candidates = []
    
    # Aligned construction
    candidates.append(-a + 0.5)
    
    # Various c values
    for c_try in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        candidates.append(-a * c_try + 0.5 * c_try)
    
    # Random perturbations of aligned
    for seed in [42, 123, 456, 789]:
        np.random.seed(seed)
        pert = np.random.normal(0, 0.05, n)
        candidates.append(-a + 0.5 + pert)
    
    # Evaluate all candidates
    best_positions = None
    best_area = np.inf
    
    for pos_try in candidates:
        if time.time() > deadline:
            break
        area_try = _compute_area(pos_try, n, 20000)
        if area_try < best_area:
            best_area = area_try
            best_positions = pos_try.copy()
    
    # Local search phase 1: coarse
    if time.time() < deadline - 60:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.03
        
        np.random.seed(42)
        for it in range(8000):
            if time.time() > deadline - 60:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.005, step_size * 0.9997)
    
    # Local search phase 2: fine
    if time.time() < deadline - 30:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.01
        
        np.random.seed(123)
        for it in range(6000):
            if time.time() > deadline - 30:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.001, step_size * 0.9995)
    
    # Subgradient polish phase
    if time.time() < deadline - 10:
        current = best_positions.copy()
        current_area = best_area
        lr = 0.01
        
        for it in range(3000):
            if time.time() > deadline - 10:
                break
            
            grad = _subgradient(current, n, 20000)
            current = current - lr * grad
            current_area = _compute_area(current, n, 20000)
            
            if current_area < best_area:
                best_area = current_area
                best_positions = current.copy()
            
            lr *= 0.999
    
    # Final micro-refinement
    if time.time() < deadline - 3:
        current = best_positions.copy()
        current_area = best_area
        step_size = 0.002
        
        np.random.seed(999)
        for it in range(2000):
            if time.time() > deadline - 3:
                break
            
            idx = np.random.randint(n)
            delta = np.random.uniform(-step_size, step_size)
            trial = current.copy()
            trial[idx] += delta
            
            trial_area = _compute_area(trial, n, 20000)
            
            if trial_area < current_area:
                current = trial
                current_area = trial_area
                if current_area < best_area:
                    best_area = current_area
                    best_positions = current.copy()
            
            step_size = max(0.0005, step_size * 0.999)
    
    # Ensure finite
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
