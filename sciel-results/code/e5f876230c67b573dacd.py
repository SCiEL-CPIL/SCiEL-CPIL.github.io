# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE_FRACTION = 0.90
_START_TIME = None


def _keich_positions(n=64):
    """Classical Keich construction for n a power of 2."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return positions


def _compute_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using quadrature at M heights.
    Uses the identity: union_length = max(L) - min(L) + w*(n-1) - sum_j max(gap_j, w)
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n
    
    L = x[:, None] + a[:, None] * y[None, :]  # (n, M)
    L_sorted = np.sort(L, axis=0)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
    
    # union = max - min + w*(n-1) - sum(max(gaps, w))
    union_lengths = (L_sorted[-1, :] - L_sorted[0, :]) + w * (n - 1) - np.sum(np.maximum(gaps, w[None, :]), axis=0)
    
    area = np.mean(union_lengths)
    return area


def _compute_area_and_subgrad(x, n=64, M=20000):
    """
    Compute area and subgradient w.r.t. x using the fast identity.
    Returns (area, subgrad) where subgrad has shape (n,).
    """
    a = (np.arange(n) + 1.0) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # (M,)
    
    L = x[:, None] + a[:, None] * y[None, :]  # (n, M)
    order = np.argsort(L, axis=0)  # (n, M)
    L_sorted = np.take_along_axis(L, order, axis=0)  # (n, M)
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]  # (n-1, M)
    
    # union = max - min + w*(n-1) - sum(max(gaps, w))
    union_lengths = (L_sorted[-1, :] - L_sorted[0, :]) + w * (n - 1) - np.sum(np.maximum(gaps, w[None, :]), axis=0)
    area = np.mean(union_lengths)
    
    # Subgradient:
    # d(union)/dL_i for each i
    # union = L_max - L_min + w*(n-1) - sum_j max(gap_j, w)
    # gap_j = L_sorted[j+1] - L_sorted[j]
    # d(union)/dL_sorted[j] = -1 if j is argmin, +1 if j is argmax
    # For each gap_j: if gap_j > w, d/dL_sorted[j] -= 1, d/dL_sorted[j+1] += 1
    #                 if gap_j < w, no contribution (clamped)
    
    dL_sorted = np.zeros_like(L_sorted)  # (n, M)
    
    # min and max contributions
    min_idx = np.argmin(L_sorted, axis=0)  # (M,)
    max_idx = np.argmax(L_sorted, axis=0)  # (M,)
    dL_sorted[min_idx, np.arange(M)] -= 1.0
    dL_sorted[max_idx, np.arange(M)] += 1.0
    
    # gap contributions
    above_w = gaps > w[None, :]  # (n-1, M)
    dL_sorted[:-1, :] -= above_w  # d/dL_sorted[j] for gap j
    dL_sorted[1:, :] += above_w   # d/dL_sorted[j+1] for gap j
    
    # Map back to original indices
    dL_orig = np.zeros_like(L)  # (n, M)
    np.put_along_axis(dL_orig, order, dL_sorted, axis=0)
    
    # d(union)/dx_i = d(union)/dL_i since L_i = x_i + a_i*y
    subgrad = dL_orig.mean(axis=1)  # (n,)
    
    return area, subgrad


def _coordinate_descent(x, n=64, M=20000, deadline=None, max_cd_iters=500):
    """
    Coordinate descent to minimize area.
    """
    best_area = _compute_area(x, n, M)
    best_x = x.copy()
    w = 1.0 / n
    
    for cd_iter in range(max_cd_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(n):
            # Try different values for x[i]
            # Current value and neighbors
            candidates = [x[i] - 0.01, x[i] - 0.005, x[i] - 0.001, x[i], 
                         x[i] + 0.001, x[i] + 0.005, x[i] + 0.01]
            
            best_cand_val = x[i]
            best_cand_area = best_area
            
            for cv in candidates:
                test_x = x.copy()
                test_x[i] = cv
                test_area = _compute_area(test_x, n, M)
                if test_area < best_cand_area:
                    best_cand_area = test_area
                    best_cand_val = cv
            
            if best_cand_val != x[i]:
                x[i] = best_cand_val
                best_area = best_cand_area
                best_x = x.copy()
                improved = True
        
        if not improved:
            break
    
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = 64
    M = 20000
    deadline = _START_TIME + 630.0 * _DEADLINE_FRACTION
    
    x = _keich_positions(n)
    best_x = x.copy()
    best_area = _compute_area(x, n, M)
    
    # Phase 1: Subgradient descent
    step = 0.02
    min_step = 1e-8
    max_iter = 100000
    no_improve_count = 0
    patience = 200
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        area, subgrad = _compute_area_and_subgrad(x, n, M)
        if area < best_area - 1e-12:
            best_area = area
            best_x = x.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if no_improve_count > patience:
            break
        
        sg_norm = np.linalg.norm(subgrad)
        if sg_norm < 1e-12:
            break
        
        direction = subgrad / sg_norm
        
        # Line search: try several step sizes
        improved = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            cand = x - step * scale * direction
            cand_area = _compute_area(cand, n, M)
            if cand_area < area:
                x = cand
                improved = True
                break
        
        if not improved:
            step *= 0.7
            if step < min_step:
                break
        else:
            if it % 100 == 0 and step < 0.02:
                step = min(0.02, step * 1.3)
    
    # Phase 2: Coordinate descent refinement
    if time.time() < deadline - 30:
        x_cd, area_cd = _coordinate_descent(x, n, M, deadline=deadline)
        if area_cd < best_area:
            best_x = x_cd.copy()
            best_area = area_cd
    
    # Final validation
    if np.any(~np.isfinite(best_x)) or best_area <= 0:
        return _keich_positions(n)
    
    return best_x


# EVOLVE-BLOCK-END
