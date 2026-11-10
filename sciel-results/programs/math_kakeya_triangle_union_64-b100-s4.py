# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants for speed
_N = 64
_M_FINE = 20000
_M_COARSE = 2000

_A = (np.arange(1, _N + 1, dtype=np.float64)) / _N
_Y_FINE = (np.arange(_M_FINE, dtype=np.float64) + 0.5) / _M_FINE
_W_FINE = (1.0 - _Y_FINE) / _N
_Y_COARSE = (np.arange(_M_COARSE, dtype=np.float64) + 0.5) / _M_COARSE
_W_COARSE = (1.0 - _Y_COARSE) / _N


def _compute_area_fine(x, n=64, M=20000):
    """Compute the union area of the n triangles using the full 20000-height quadrature."""
    a = _A[:n]
    y = _Y_FINE[:M]
    w = _W_FINE[:M]
    L = x[:, None] + a[:, None] * y[None, :]
    Ls = np.sort(L, axis=0)
    gaps = Ls[1:, :] - Ls[:-1, :]
    gaps = np.minimum(gaps, w[None, :])
    union_len = w + gaps.sum(axis=0)
    area = union_len.mean()
    return area


def _compute_area_coarse(x, n=64, M=2000):
    """Compute the union area using a coarse 2000-height grid (for speed during search)."""
    a = _A[:n]
    y = _Y_COARSE[:M]
    w = _W_COARSE[:M]
    L = x[:, None] + a[:, None] * y[None, :]
    Ls = np.sort(L, axis=0)
    gaps = Ls[1:, :] - Ls[:-1, :]
    gaps = np.minimum(gaps, w[None, :])
    union_len = w + gaps.sum(axis=0)
    area = union_len.mean()
    return area


def _compute_subgradient(x, n=64, M=2000):
    """
    Compute a subgradient of the coarse-grid union area w.r.t. x.
    
    The union length at height m is:
        U_m = w_m + sum_{j=0}^{n-2} min(L_{(j+1),m} - L_{(j),m}, w_m)
    where L_{(j),m} is the j-th smallest of {x_i + a_i * y_m}.
    
    dU_m/dx_i = 1 if i is the largest index in the sorted order at height m
               -1 if i is the smallest index in the sorted order at height m
               0 otherwise, for each gap that is NOT capped by w_m.
    
    More precisely, for each consecutive pair in sorted order:
      If gap < w_m: the pair contributes (d/dx of (L_{j+1} - L_j)) = e_{j+1} - e_j
      If gap >= w_m: the pair contributes 0 (capped)
    Plus the base w_m which doesn't depend on x.
    
    So subgradient component for x_i = sum over all heights m of:
      +1 for each gap where i is the upper element and gap < w_m
      -1 for each gap where i is the lower element and gap < w_m
    Then divide by M to get the mean.
    """
    a = _A[:n]
    y = _Y_COARSE[:M]
    w = _W_COARSE[:M]
    
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
    
    # Get sorted indices
    idx = np.argsort(L, axis=0)  # shape (n, M), idx[j, m] = index of j-th smallest at height m
    
    # Get sorted values
    Ls = np.take_along_axis(L, idx, axis=0)  # shape (n, M)
    
    # Compute gaps
    gaps = Ls[1:, :] - Ls[:-1, :]  # shape (n-1, M)
    
    # Mask: which gaps are NOT capped (i.e., gap < w_m)
    not_capped = gaps < w[None, :]  # shape (n-1, M)
    
    # For each uncapped gap at position j (between sorted positions j and j+1),
    # the subgradient contribution is: +1 to x_{idx[j+1,m]}, -1 to x_{idx[j,m]}
    
    # Build subgradient array
    grad = np.zeros((n, M), dtype=np.float64)
    
    # For the lower element of each uncapped gap: -1
    lower_idx = idx[:-1, :]  # shape (n-1, M)
    # For the upper element of each uncapped gap: +1
    upper_idx = idx[1:, :]   # shape (n-1, M)
    
    # Use scatter-add via bincount or direct indexing
    # Flatten for efficiency
    lower_flat = lower_idx.ravel()
    upper_flat = upper_idx.ravel()
    mask_flat = not_capped.ravel()
    
    # Only count where not capped
    lower_valid = lower_flat[mask_flat]
    upper_valid = upper_flat[mask_flat]
    m_idx = np.repeat(np.arange(M), n - 1)[mask_flat]  # which height each gap belongs to
    
    # Accumulate
    grad_flat = np.zeros(n * M, dtype=np.float64)
    np.add.at(grad_flat, lower_valid * M + m_idx, -1.0)
    np.add.at(grad_flat, upper_valid * M + m_idx, 1.0)
    
    grad = grad_flat.reshape(n, M) / M  # Divide by M for mean
    
    # Sum over heights to get gradient w.r.t. each x_i
    subgrad = grad.sum(axis=1)
    
    return subgrad


def _keich_init(n=64):
    """Classical Keich construction for n a power of 2."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1, dtype=np.float64)
    positions = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(bin(i)[2:]), dtype=np.float64)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions[i] = np.sum((1.0 - j) * eps * (2.0 ** -j) / k)
    return positions


def _subgradient_method(x, n=64, max_iters=500, deadline=None):
    """
    Subgradient method for minimizing the coarse-grid union area.
    Uses diminishing step sizes and momentum.
    """
    best_x = x.copy()
    best_area = _compute_area_coarse(x, n)
    current_area = best_area
    
    # Momentum
    v = np.zeros(n, dtype=np.float64)
    mu = 0.9  # momentum coefficient
    
    # Step size schedule
    step0 = 0.05
    decay = 0.995
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute subgradient
        g = _compute_subgradient(x, n)
        
        # Update with momentum
        v = mu * v - step0 * (g / (np.linalg.norm(g) + 1e-12))
        x = x + v
        
        # Evaluate
        area = _compute_area_coarse(x, n)
        
        if area < best_area:
            best_area = area
            best_x = x.copy()
        
        # Decay step size
        step0 *= decay
        
        if step0 < 1e-10:
            break
    
    return best_x, best_area


def _smooth_min_subgradient(x, n=64, max_iters=300, deadline=None):
    """
    Smoothed subgradient method using log-sum-exp approximation for the min in gaps.
    This provides a smoother objective that can be optimized more effectively.
    """
    best_x = x.copy()
    best_area = _compute_area_coarse(x, n)
    
    # Temperature for soft-min
    tau = 0.01
    tau_decay = 0.99
    
    step0 = 0.03
    step_decay = 0.998
    v = np.zeros(n, dtype=np.float64)
    mu = 0.85
    
    a = _A[:n]
    y = _Y_COARSE
    w = _W_COARSE
    M = len(y)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        L = x[:, None] + a[:, None] * y[None, :]
        Ls = np.sort(L, axis=0)
        gaps = Ls[1:, :] - Ls[:-1, :]
        
        # Soft-min: min(gap, w) ≈ -tau * logsumexp(-gap/tau, -w/tau)
        # But we need the gradient of the union area, which involves the soft-min.
        # The gradient of -tau * log(exp(-gap/tau) + exp(-w/tau)) w.r.t. gap
        # = exp(-gap/tau) / (exp(-gap/tau) + exp(-w/tau))
        
        # Compute soft-min weight for each gap
        # weight = exp(-gap/tau) / (exp(-gap/tau) + exp(-w/tau))
        # This is 1 when gap << w, 0 when gap >> w
        
        gap_exp = np.exp(-gaps / tau)
        w_exp = np.exp(-w[None, :] / tau)
        weights = gap_exp / (gap_exp + w_exp)  # shape (n-1, M)
        
        # Now compute subgradient using these weights
        idx = np.argsort(L, axis=0)
        lower_idx = idx[:-1, :].ravel()
        upper_idx = idx[1:, :].ravel()
        m_idx = np.repeat(np.arange(M), n - 1)
        w_flat = weights.ravel()
        
        grad_flat = np.zeros(n * M, dtype=np.float64)
        np.add.at(grad_flat, lower_idx * M + m_idx, -w_flat)
        np.add.at(grad_flat, upper_idx * M + m_idx, w_flat)
        
        grad = grad_flat.reshape(n, M).sum(axis=1) / M
        
        # Update
        norm_g = np.linalg.norm(grad)
        v = mu * v - step0 * (grad / (norm_g + 1e-12))
        x = x + v
        
        area = _compute_area_coarse(x, n)
        if area < best_area:
            best_area = area
            best_x = x.copy()
        
        # Decay
        step0 *= step_decay
        tau *= tau_decay
        
        if step0 < 1e-10:
            break
    
    return best_x, best_area


def _coordinate_descent(x, n=64, max_iters=200, tol=1e-12, deadline=None):
    """
    Coordinate descent on the coarse-grid area.
    """
    best_x = x.copy()
    best_area = _compute_area_coarse(x, n)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best_x, best_area
            base = best_x[i]
            steps = np.concatenate([
                np.linspace(-0.05, 0.05, 21),
                np.linspace(-0.2, -0.05, 10),
                np.linspace(0.05, 0.2, 10),
                np.linspace(-0.5, -0.2, 5),
                np.linspace(0.2, 0.5, 5),
            ])
            best_local = base
            best_local_area = best_area
            for s in steps:
                trial = best_x.copy()
                trial[i] = base + s
                ar = _compute_area_coarse(trial, n)
                if ar < best_local_area - 1e-15:
                    best_local_area = ar
                    best_local = base + s
            if abs(best_local - base) > 1e-14:
                best_x[i] = best_local
                best_area = best_local_area
                improved = True
        if not improved:
            break
    return best_x, best_area


def _local_search(x, n=64, max_iters=300, deadline=None):
    """
    Randomized local search with adaptive step sizes.
    """
    best_x = x.copy()
    best_area = _compute_area_coarse(x, n)
    step_size = 0.01

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        i = np.random.randint(n)
        candidates = best_x[i] + step_size * np.random.randn(8)
        best_cand = best_x[i]
        best_cand_area = best_area
        for c in candidates:
            trial = best_x.copy()
            trial[i] = c
            ar = _compute_area_coarse(trial, n)
            if ar < best_cand_area - 1e-15:
                best_cand_area = ar
                best_cand = c
        if abs(best_cand - best_x[i]) > 1e-14:
            best_x[i] = best_cand
            best_area = best_cand_area
            if it % 50 == 49:
                step_size *= 0.8
        else:
            step_size *= 0.95
            if step_size < 1e-8:
                break
    return best_x, best_area


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns 64 finite floats x_0..x_63.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 620  # Safety margin

    # Initialize with Keich construction
    x = _keich_init(n)
    best_x = x.copy()
    best_area = _compute_area_coarse(x, n)

    # Phase 1: Subgradient method (exploits convexity)
    if time.time() < deadline - 120:
        x, area = _subgradient_method(x, n, max_iters=400, deadline=deadline - 120)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 2: Smoothed subgradient with decreasing temperature
    if time.time() < deadline - 60:
        x, area = _smooth_min_subgradient(best_x, n, max_iters=300, deadline=deadline - 60)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 3: Coordinate descent polish
    if time.time() < deadline - 30:
        x, area = _coordinate_descent(best_x, n, max_iters=100, deadline=deadline - 30)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 4: Final local search
    if time.time() < deadline - 10:
        x, area = _local_search(best_x, n, max_iters=200, deadline=deadline - 10)
        if area < best_area:
            best_x, best_area = x, area

    # Verify with fine grid
    fine_area = _compute_area_fine(best_x, n)
    if not np.isfinite(fine_area) or fine_area <= 0:
        return _keich_init(n)

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
