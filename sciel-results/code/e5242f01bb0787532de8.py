# EVOLVE-BLOCK-START
import numpy as np
import time

# Precompute constants for speed
_N = 64
_M_FINE = 20000
_M_COARSE = 5000

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


def _compute_area_coarse(x, n=64, M=5000):
    """Compute the union area using a coarse 5000-height grid (for speed during search)."""
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


def _compute_subgradient(x, n=64, M=5000):
    """
    Compute a subgradient of the coarse-grid union area w.r.t. x.
    """
    a = _A[:n]
    y = _Y_COARSE[:M]
    w = _W_COARSE[:M]
    
    L = x[:, None] + a[:, None] * y[None, :]  # shape (n, M)
    
    # Get sorted indices
    idx = np.argsort(L, axis=0)  # shape (n, M)
    
    # Get sorted values
    Ls = np.take_along_axis(L, idx, axis=0)  # shape (n, M)
    
    # Compute gaps
    gaps = Ls[1:, :] - Ls[:-1, :]  # shape (n-1, M)
    
    # Mask: which gaps are NOT capped (i.e., gap < w_m)
    not_capped = gaps < w[None, :]  # shape (n-1, M)
    
    # Build subgradient array
    grad = np.zeros((n, M), dtype=np.float64)
    
    lower_idx = idx[:-1, :]  # shape (n-1, M)
    upper_idx = idx[1:, :]   # shape (n-1, M)
    
    lower_flat = lower_idx.ravel()
    upper_flat = upper_idx.ravel()
    mask_flat = not_capped.ravel()
    
    lower_valid = lower_flat[mask_flat]
    upper_valid = upper_flat[mask_flat]
    m_idx = np.repeat(np.arange(M), n - 1)[mask_flat]
    
    grad_flat = np.zeros(n * M, dtype=np.float64)
    np.add.at(grad_flat, lower_valid * M + m_idx, -1.0)
    np.add.at(grad_flat, upper_valid * M + m_idx, 1.0)
    
    grad = grad_flat.reshape(n, M) / M
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


def _subgradient_method_v2(x, n=64, max_iters=1000, deadline=None):
    """
    Improved subgradient method with:
    - Finer coarse grid (5000 heights)
    - Diminishing step size: step_k = c / sqrt(k)
    - Nesterov acceleration
    """
    best_x = x.copy()
    best_area = _compute_area_coarse(x, n)
    
    a = _A[:n]
    y = _Y_COARSE
    w = _W_COARSE
    M = len(y)
    
    # Nesterov acceleration
    y_curr = x.copy()
    v = np.zeros(n, dtype=np.float64)
    
    # Step size parameters
    c = 0.02  # step size constant
    
    for k in range(1, max_iters + 1):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute subgradient at y_curr
        g = _compute_subgradient(y_curr, n, M)
        
        # Step size
        step_k = c / np.sqrt(k)
        
        # Update
        y_new = y_curr - step_k * g
        v_new = (k - 1) / (k + 2) * (y_new - x)
        x = y_new + v_new
        
        # Evaluate at x
        area = _compute_area_coarse(x, n)
        
        if area < best_area:
            best_area = area
            best_x = x.copy()
        
        y_curr = x.copy()
    
    return best_x, best_area


def _projected_subgradient(x, n=64, max_iters=800, deadline=None):
    """
    Projected subgradient with Polyak step size.
    Projects x to a bounded box to prevent divergence.
    """
    best_x = x.copy()
    best_area = _compute_area_coarse(x, n)
    
    # Bound for x values (triangles should stay in reasonable range)
    x_min = -1.0
    x_max = 1.0
    
    y_curr = x.copy()
    v = np.zeros(n, dtype=np.float64)
    mu = 0.85  # momentum
    
    # Polyak step size: need an upper bound on optimal area
    # The minimum possible area is at least w_mean = mean((1-y)/64) = 1/128
    # A reasonable upper bound from initial solution
    area_upper = best_area
    area_lower = 1.0 / 128.0  # theoretical lower bound
    
    for k in range(1, max_iters + 1):
        if deadline is not None and time.time() > deadline:
            break
        
        g = _compute_subgradient(y_curr, n)
        
        # Polyak step size
        area_curr = _compute_area_coarse(y_curr, n)
        step_k = (area_curr - area_lower) / (np.linalg.norm(g) ** 2 + 1e-12)
        step_k = min(step_k, 0.05)  # Cap step size
        
        # Momentum update
        v = mu * v - step_k * g
        y_new = y_curr + v
        
        # Project to box
        y_new = np.clip(y_new, x_min, x_max)
        
        area = _compute_area_coarse(y_new, n)
        
        if area < best_area:
            best_area = area
            best_x = y_new.copy()
        
        y_curr = y_new.copy()
        
        # Update upper bound
        if area_curr < area_upper:
            area_upper = area_curr
    
    return best_x, best_area


def _coordinate_descent_fine(x, n=64, max_iters=50, tol=1e-12, deadline=None):
    """
    Coordinate descent on the FINE-grid area for final polish.
    """
    best_x = x.copy()
    best_area = _compute_area_fine(x, n)

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best_x, best_area
            base = best_x[i]
            # Golden section search around current value
            best_local = base
            best_local_area = best_area
            # Try a range of offsets
            for scale in [0.01, 0.02, 0.05, 0.1, 0.2]:
                for sign in [-1, 1]:
                    trial = best_x.copy()
                    trial[i] = base + sign * scale
                    ar = _compute_area_fine(trial, n)
                    if ar < best_local_area - 1e-15:
                        best_local_area = ar
                        best_local = base + sign * scale
            if abs(best_local - base) > 1e-14:
                best_x[i] = best_local
                best_area = best_local_area
                improved = True
        if not improved:
            break
    return best_x, best_area


def _local_search_fine(x, n=64, max_iters=100, deadline=None):
    """
    Randomized local search with fine-grid evaluation.
    """
    best_x = x.copy()
    best_area = _compute_area_fine(x, n)
    step_size = 0.005

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        i = np.random.randint(n)
        candidates = best_x[i] + step_size * np.random.randn(6)
        best_cand = best_x[i]
        best_cand_area = best_area
        for c in candidates:
            trial = best_x.copy()
            trial[i] = c
            ar = _compute_area_fine(trial, n)
            if ar < best_cand_area - 1e-15:
                best_cand_area = ar
                best_cand = c
        if abs(best_cand - best_x[i]) > 1e-14:
            best_x[i] = best_cand
            best_area = best_cand_area
            if it % 30 == 29:
                step_size *= 0.85
        else:
            step_size *= 0.97
            if step_size < 1e-9:
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

    # Phase 1: Nesterov-accelerated subgradient (coarse grid)
    if time.time() < deadline - 180:
        x, area = _subgradient_method_v2(x, n, max_iters=1000, deadline=deadline - 180)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 2: Projected subgradient with Polyak step (coarse grid)
    if time.time() < deadline - 120:
        x, area = _projected_subgradient(best_x, n, max_iters=800, deadline=deadline - 120)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 3: Coordinate descent on fine grid
    if time.time() < deadline - 40:
        x, area = _coordinate_descent_fine(best_x, n, max_iters=50, deadline=deadline - 40)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 4: Final local search on fine grid
    if time.time() < deadline - 10:
        x, area = _local_search_fine(best_x, n, max_iters=100, deadline=deadline - 10)
        if area < best_area:
            best_x, best_area = x, area

    # Verify with fine grid
    fine_area = _compute_area_fine(best_x, n)
    if not np.isfinite(fine_area) or fine_area <= 0:
        return _keich_init(n)

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
