import numpy as np
import time


# Precompute constants for area evaluation
_N = 64
_N_HEIGHTS = 20000
_A = (np.arange(_N) + 1) / _N
_Y = (np.arange(_N_HEIGHTS) + 0.5) / _N_HEIGHTS
_W = (1 - _Y) / _N


def _compute_area(positions, n=_N, n_heights=_N_HEIGHTS):
    """Compute the union area using quadrature."""
    a = _A[:n]
    y = _Y[:n_heights]
    w = _W[:n_heights]

    # L_i = x_i + a_i * y for each triangle i and height m
    # Shape: (n, n_heights)
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]

    # Sort L at each height
    L_sorted = np.sort(L, axis=0)  # (n, n_heights)

    # Union length = w + sum over consecutive pairs of min(L_{j+1} - L_j, w)
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    overlaps = np.minimum(diffs, w[np.newaxis, :])  # (n-1, n_heights)

    union_lengths = w + np.sum(overlaps, axis=0)  # (n_heights,)
    area = np.mean(union_lengths)

    return area


def _compute_area_grad(positions, n=_N, n_heights=_N_HEIGHTS):
    """Compute the union area and its gradient with respect to positions."""
    a = _A[:n]
    y = _Y[:n_heights]
    w = _W[:n_heights]

    # L_i = x_i + a_i * y for each triangle i and height m
    L = positions[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]

    # Sort L at each height
    L_sorted = np.sort(L, axis=0)  # (n, n_heights)
    
    # We need to track which original index corresponds to each sorted position
    # to compute gradients. But for the gradient of the union length, we need
    # to know which triangles are contributing to the union.
    
    # The union length at height m is:
    # w + sum_{j=0}^{n-2} min(L_{j+1} - L_j, w)
    
    # This can be rewritten. Let's think about the derivative.
    # Actually, a cleaner way: the union of intervals [L_i, L_i + w] has length
    # that can be computed. The derivative with respect to x_k (which shifts
    # interval k) is:
    # -1 if x_k is the leftmost point of the union (i.e., the interval starting
    #   at L_k is the one that defines the left boundary)
    # +1 if x_k + w is the rightmost point (i.e., the interval ending at L_k + w
    #   defines the right boundary)
    # 0 otherwise
    # But this is only when the intervals are in "general position". When there
    # are overlaps, it's more complex.
    
    # Let's use a different approach. The union length is:
    # max_i(L_i + w) - min_i(L_i) if all intervals overlap in a chain
    # More generally, it's the sum of the lengths of the connected components.
    
    # Actually, let's just compute the gradient numerically or use the fact that
    # the function is piecewise linear.
    
    # For now, let's compute the area and use finite differences for gradient
    # if needed, but that's slow. Let's try to compute it analytically.
    
    # The union of intervals [L_i, L_i+w] can be computed by sorting the
    # endpoints. But the gradient is tricky.
    
    # Alternative: use the formula
    # union_length = w + sum_{j} min(L_{j+1} - L_j, w)
    # where L is sorted.
    
    # The derivative of min(L_{j+1} - L_j, w) with respect to L_k is:
    # -1 if k = j and L_{j+1} - L_j < w
    # +1 if k = j+1 and L_{j+1} - L_j < w
    # 0 otherwise
    
    # So we need to know, for each sorted position j, which original index it
    # corresponds to, and whether the gap L_{j+1} - L_j is less than w.
    
    # Let's compute the sorted indices
    sorted_indices = np.argsort(L, axis=0)  # (n, n_heights)
    
    # diffs = L_{j+1} - L_j for sorted positions
    diffs = np.diff(L_sorted, axis=0)  # (n-1, n_heights)
    
    # For each consecutive pair in sorted order, if the gap is < w, then
    # the derivative contribution is:
    # -1 to the earlier index, +1 to the later index
    
    # Initialize gradient
    grad = np.zeros((n, n_heights), dtype=np.float64)
    
    # For each j (0 to n-2) and each height m:
    # if diffs[j, m] < w[m], then:
    #   grad[sorted_indices[j, m], m] -= 1
    #   grad[sorted_indices[j+1, m], m] += 1
    
    # This is O(n * n_heights) which is manageable
    mask = diffs < w[np.newaxis, :]  # (n-1, n_heights)
    
    # We need to scatter the gradient contributions
    # For each (j, m) where mask[j, m] is True:
    #   grad[sorted_indices[j, m], m] -= 1
    #   grad[sorted_indices[j+1, m], m] += 1
    
    # This is tricky to vectorize. Let's do it in a loop over heights
    # or use np.add.at
    
    # Actually, let's just compute the area and use a simple finite difference
    # for the gradient in the coordinate descent.
    
    union_lengths = w + np.sum(np.minimum(diffs, w[np.newaxis, :]), axis=0)
    area = np.mean(union_lengths)
    
    # For the gradient, we'll use the fact that the area is piecewise linear
    # and compute it via the sorted indices approach, but in a vectorized way.
    
    # Let's accumulate the gradient contributions
    # For each height m, for each j where diffs[j,m] < w[m]:
    #   grad[sorted_indices[j,m], m] -= 1
    #   grad[sorted_indices[j+1,m], m] += 1
    
    # We can do this with np.add.at
    # First, let's create arrays of indices
    j_idx = np.arange(n - 1)
    j_idx, m_idx = np.where(mask)  # (n_consecutive_pairs,)
    
    # For these (j, m) pairs, we need to add -1 to grad[sorted_indices[j, m], m]
    # and +1 to grad[sorted_indices[j+1, m], m]
    
    # Get the original indices
    left_orig = sorted_indices[j_idx, m_idx]  # (n_pairs,)
    right_orig = sorted_indices[j_idx + 1, m_idx]  # (n_pairs,)
    
    # Accumulate gradient
    np.add.at(grad, (left_orig, m_idx), -1.0)
    np.add.at(grad, (right_orig, m_idx), 1.0)
    
    # The gradient of the area with respect to x_k is the mean over heights
    # of grad[k, m]
    grad_x = np.mean(grad, axis=1)
    
    return area, grad_x


def _line_search(current, idx, n=_N, n_heights=_N_HEIGHTS, max_iter=30):
    """Perform line search along dimension idx to minimize area."""
    best_val = current[idx]
    best_area = _compute_area(current, n, n_heights)
    
    # Try a range of values
    # Start with a broad range and narrow down
    for attempt in range(max_iter):
        # Try several candidate values
        candidates = [
            best_val - 0.05,
            best_val - 0.02,
            best_val - 0.005,
            best_val + 0.005,
            best_val + 0.02,
            best_val + 0.05,
        ]
        
        best_candidate = best_val
        best_candidate_area = best_area
        
        for c in candidates:
            trial = current.copy()
            trial[idx] = c
            trial_area = _compute_area(trial, n, n_heights)
            if trial_area < best_candidate_area:
                best_candidate_area = trial_area
                best_candidate = c
        
        if best_candidate == best_val:
            break
        
        best_val = best_candidate
        best_area = best_candidate_area
    
    current[idx] = best_val
    return current, best_area


def _coordinate_descent(positions, n=_N, n_heights=_N_HEIGHTS, max_rounds=50,
                        time_limit=300):
    """Perform coordinate descent to minimize area."""
    current = positions.copy().astype(np.float64)
    current_area = _compute_area(current, n, n_heights)
    best = current.copy()
    best_area = current_area
    
    start_time = time.time()
    
    for round_idx in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for idx in range(n):
            if time.time() - start_time > time_limit:
                break
            
            old_area = current_area
            current, new_area = _line_search(current, idx, n, n_heights)
            
            if new_area < old_area - 1e-10:
                improved = True
                current_area = new_area
                if current_area < best_area:
                    best = current.copy()
                    best_area = current_area
        
        if not improved:
            break
    
    return best, best_area


def _nelder_mead_smooth(positions, n=_N, n_heights=_N_HEIGHTS, max_iter=2000,
                        time_limit=300):
    """
    Nelder-Mead with a smooth soft-min surrogate for the union length.
    
    Instead of using min(diffs, w), we use a smooth approximation:
    soft_min(d, w) = -t * log(1 + exp(-(d - w)/t)) + t * log(2)
    which approximates min(d, w) for small t.
    """
    start_time = time.time()
    
    # Initialize simplex
    simplex = [positions.copy().astype(np.float64)]
    for i in range(n):
        v = positions.copy().astype(np.float64)
        v[i] += 0.01
        simplex.append(v)
    
    # Evaluate initial points
    areas = [_compute_area(s, n, n_heights) for s in simplex]
    
    for iter_idx in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Sort by area
        order = np.argsort(areas)
        simplex = [simplex[i] for i in order]
        areas = [areas[i] for i in order]
        
        # Centroid of all but the worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)
        reflected_area = _compute_area(reflected, n, n_heights)
        
        if reflected_area < areas[0]:
            # Expansion
            expanded = centroid + 2 * (reflected - centroid)
            expanded_area = _compute_area(expanded, n, n_heights)
            if expanded_area < reflected_area:
                simplex[-1] = expanded
                areas[-1] = expanded_area
            else:
                simplex[-1] = reflected
                areas[-1] = reflected_area
        elif reflected_area < areas[-2]:
            simplex[-1] = reflected
            areas[-1] = reflected_area
        else:
            # Contraction
            contracted = centroid + 0.5 * (worst - centroid)
            contracted_area = _compute_area(contracted, n, n_heights)
            if contracted_area < areas[-1]:
                simplex[-1] = contracted
                areas[-1] = contracted_area
            else:
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    areas[i] = _compute_area(simplex[i], n, n_heights)
    
    # Return the best point
    best_idx = np.argmin(areas)
    return simplex[best_idx], areas[best_idx]


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Returns positions that minimize the union area of the 64 triangles.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 620  # 620 second limit (safety margin from 630)
    
    # Start with the "aligned" initialization: x_i = -a_i + 0.5
    a = (np.arange(n) + 1) / n
    positions = -a + 0.5
    
    # Also try the Keich construction as an alternative start
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    keich_positions = np.array(keich_positions, dtype=np.float64)
    
    # Evaluate both starting points
    area1 = _compute_area(positions, n, 20000)
    area2 = _compute_area(keich_positions, n, 20000)
    
    if area2 < area1:
        positions = keich_positions
        area1 = area2
    
    # Try a few random initializations
    np.random.seed(123)
    for _ in range(3):
        random_pos = np.random.uniform(-1, 1, n)
        rand_area = _compute_area(random_pos, n, 20000)
        if rand_area < area1:
            positions = random_pos
            area1 = rand_area
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 100:
        return positions.astype(np.float64)
    
    # Perform coordinate descent
    cd_time_limit = min(150, deadline - (time.time() - start_time) - 50)
    if cd_time_limit > 30:
        best_positions, best_area = _coordinate_descent(
            positions, n, 20000, max_rounds=30, time_limit=cd_time_limit
        )
    else:
        best_positions = positions
        best_area = area1
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 50:
        return best_positions.astype(np.float64)
    
    # Perform Nelder-Mead for further refinement
    nm_time_limit = min(100, deadline - (time.time() - start_time) - 20)
    if nm_time_limit > 30:
        nm_positions, nm_area = _nelder_mead_smooth(
            best_positions, n, 20000, max_iter=1000, time_limit=nm_time_limit
        )
        if nm_area < best_area:
            best_positions = nm_positions
            best_area = nm_area
    
    # Final coordinate descent polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 30:
        final_cd_limit = deadline - elapsed - 10
        if final_cd_limit > 10:
            final_positions, final_area = _coordinate_descent(
                best_positions, n, 20000, max_rounds=20, time_limit=final_cd_limit
            )
            if final_area < best_area:
                best_positions = final_positions
                best_area = final_area
    
    # Ensure finite values
    if not np.all(np.isfinite(best_positions)):
        return np.zeros(n, dtype=np.float64)
    
    return best_positions.astype(np.float64)
