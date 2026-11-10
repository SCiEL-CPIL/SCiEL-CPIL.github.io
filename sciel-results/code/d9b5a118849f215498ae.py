# EVOLVE-BLOCK-START
import numpy as np
import time


def _area_fast(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles using vectorized quadrature."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    union_lengths = w[np.newaxis, :] + np.minimum(gaps, w[np.newaxis, :]).sum(axis=0)
    area = np.mean(union_lengths)
    return area


def _subgrad_xi(x, i, n=64, n_heights=20000):
    """Compute subgradient of area with respect to x_i."""
    a = (np.arange(n) + 1) / n
    y = (np.arange(n_heights) + 0.5) / n_heights
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # For each height, find the rank of L_i in the sorted order
    # and compute the subgradient contribution
    
    # The union length U = w + sum_j min(L_{j+1} - L_j, w)
    # where L_j are sorted.
    
    # dU/dL_k (where L_k is the k-th smallest) is:
    # Let's think of it as: U = w + sum_{j=0}^{n-2} min(L_{j+1} - L_j, w)
    
    # For a given height, suppose L_i has rank r in the sorted order.
    # The term involving L_i appears in:
    # - min(L_r - L_{r-1}, w) if r > 0: derivative w.r.t. L_r is -1 if L_r - L_{r-1} < w, else 0
    # - min(L_{r+1} - L_r, w) if r < n-1: derivative w.r.t. L_r is +1 if L_{r+1} - L_r < w, else 0
    # Wait, that's not right. Let me reconsider.
    
    # U = w + sum_{j=0}^{n-2} min(L_{j+1} - L_j, w)
    # dU/dL_r = d/dL_r [min(L_r - L_{r-1}, w)] + d/dL_r [min(L_{r+1} - L_r, w)]
    #        = (-1 if L_r - L_{r-1} < w else 0) + (-1 if L_{r+1} - L_r < w else 0)
    # Wait no: d/dL_r [min(L_r - L_{r-1}, w)] = 1 if L_r - L_{r-1} < w else 0
    #        d/dL_r [min(L_{r+1} - L_r, w)] = -1 if L_{r+1} - L_r < w else 0
    
    # So dU/dL_r = [L_r - L_{r-1} < w] - [L_{r+1} - L_r < w]
    
    # And dU/dx_i = mean_y dU/dL_i(y) * dL_i(y)/dx_i = mean_y dU/dL_i(y) * 1
    
    # Compute for each height:
    # 1. Sort L and find rank of L_i
    # 2. Compute the subgradient
    
    # Vectorized approach:
    # For each height, we need the rank of L_i(y) among all L_j(y)
    
    # Sort L and track original indices
    sort_idx = np.argsort(L, axis=0)  # shape (n, n_heights)
    
    # Find rank of i at each height
    # rank[i, m] = number of j such that L[j, m] < L[i, m]
    # We can compute this as: for each height m, count how many sort_idx[k, m] < i... no
    # Better: rank[i, m] = sum_j [L[j, m] < L[i, m]]
    
    # Use broadcasting: L[j, m] < L[i, m] for all j, m
    # L_i_expanded = L[i, :] shape (n_heights,)
    # L_all = L shape (n, n_heights)
    # less = L_all < L_i_expanded[np.newaxis, :] shape (n, n_heights)
    # rank = less.sum(axis=0) shape (n_heights,)
    
    L_i = L[i, :]  # shape (n_heights,)
    less = L < L_i[np.newaxis, :]  # shape (n, n_heights)
    rank = less.sum(axis=0)  # shape (n_heights,), rank[i, m] = number of L_j < L_i at height m
    
    # Now compute subgradient at each height
    # For height m, if rank[i, m] = r:
    # dU/dL_i = [L_{r} - L_{r-1} < w] - [L_{r+1} - L_r < w] where L_r is L_i
    # i.e., [L_i - L_{prev} < w] - [L_{next} - L_i < w]
    
    # Get the sorted L values
    L_sorted = np.sort(L, axis=0)  # shape (n, n_heights)
    
    # For each height, get L_prev and L_next relative to L_i
    # L_prev at rank r is L_sorted[r-1, m] if r > 0, else -inf
    # L_next at rank r is L_sorted[r+1, m] if r < n-1, else +inf
    
    # Build arrays for L_prev and L_next
    # rank shape (n_heights,), L_sorted shape (n, n_heights)
    # We need L_sorted[rank-1, m] and L_sorted[rank+1, m]
    
    # Use advanced indexing
    m_idx = np.arange(n_heights)
    r = rank  # shape (n_heights,)
    
    # L_prev: for each m, L_sorted[r[m]-1, m] if r[m] > 0
    L_prev = np.where(r > 0, L_sorted[np.clip(r - 1, 0, n-1), m_idx], -np.inf)
    L_next = np.where(r < n-1, L_sorted[np.clip(r + 1, 0, n-1), m_idx], np.inf)
    
    # Subgradient at each height
    # dU/dL_i = [L_i - L_prev < w] - [L_next - L_i < w]
    sg = (L_i - L_prev < w).astype(float) - (L_next - L_i < w).astype(float)
    
    # Mean over heights
    subgrad = np.mean(sg)
    return subgrad


def _keich_initial(n=64):
    """Generate the Keich construction initial positions."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Optimize Kakeya needle triangle positions to minimize union area.
    Uses coordinate descent with subgradient-based 1D optimization.
    """
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin
    
    n = 64
    x = _keich_initial(n)
    best_x = x.copy()
    best_area = _area_fast(x, n)
    
    # Coordinate descent
    max_passes = 20
    tol = 1e-12
    
    for pass_num in range(max_passes):
        if time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            # 1D optimization for coordinate i using subgradient bisection
            # The area is convex in x_i, so we can use subgradient to find minimum
            
            # Find a bracket for the optimal x_i
            # Start with current x[i] and expand
            lo = x[i] - 0.1
            hi = x[i] + 0.1
            
            # Evaluate subgradient at lo and hi
            sg_lo = _subgrad_xi(x, i, n)
            # We need to adjust x[i] to lo and evaluate
            x[i] = lo
            sg_lo = _subgrad_xi(x, i, n)
            x[i] = hi
            sg_hi = _subgrad_xi(x, i, n)
            
            # If sg_lo > 0, minimum is to the left of lo
            # If sg_hi < 0, minimum is to the right of hi
            # Otherwise, minimum is in [lo, hi]
            
            # Bisection on subgradient
            for _ in range(30):  # 30 iterations of bisection
                if time.time() > deadline:
                    break
                
                mid = (lo + hi) / 2
                x[i] = mid
                sg_mid = _subgrad_xi(x, i, n)
                
                if sg_mid > 0:
                    hi = mid
                else:
                    lo = mid
            
            # Set x[i] to the midpoint
            x[i] = (lo + hi) / 2
            new_area = _area_fast(x, n)
            
            if new_area < best_area - 1e-15:
                best_area = new_area
                best_x = x.copy()
                improved = True
            else:
                # Revert if no improvement
                x[i] = best_x[i]
        
        if not improved:
            break
    
    # Return best result
    final_area = _area_fast(best_x, n)
    if np.isfinite(final_area) and final_area > 0:
        return best_x
    return _keich_initial(n)


# EVOLVE-BLOCK-END
