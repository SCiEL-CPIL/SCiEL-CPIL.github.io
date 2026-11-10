# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_area_fine(x, n=64, M=20000):
    """Compute the union area of the n triangles using the full 20000-height quadrature."""
    a = (np.arange(1, n + 1, dtype=np.float64)) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    # L shape (n, M): L[i, m] = x[i] + a[i]*y[m]
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) / n  # shape (1, M)
    # Sort along axis 0
    Ls = np.sort(L, axis=0)
    # Union length = w + sum over consecutive pairs of min(L_{j+1}-L_j, w)
    gaps = Ls[1:, :] - Ls[:-1, :]  # shape (n-1, M)
    gaps = np.minimum(gaps, w)
    union_len = w[0, :] + gaps.sum(axis=0)  # shape (M,)
    area = union_len.mean()
    return area


def _compute_area_coarse(x, n=64, M=2000):
    """Compute the union area using a coarse 2000-height grid (for speed during search)."""
    a = (np.arange(1, n + 1, dtype=np.float64)) / n
    y = (np.arange(M, dtype=np.float64) + 0.5) / M
    L = x[:, None] + a[:, None] * y[None, :]
    w = (1.0 - y[None, :]) / n
    Ls = np.sort(L, axis=0)
    gaps = Ls[1:, :] - Ls[:-1, :]
    gaps = np.minimum(gaps, w)
    union_len = w[0, :] + gaps.sum(axis=0)
    area = union_len.mean()
    return area


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


def _coordinate_descent(x, n=64, max_iters=200, tol=1e-12, deadline=None):
    """
    Coordinate descent on the coarse-grid area.
    For each coordinate x_i, try a small set of candidate values around the current
    one and pick the best.
    """
    current_area = _compute_area_coarse(x, n)
    best_x = x.copy()
    best_area = current_area

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                return best_x, best_area
            # Try a grid of perturbations around x[i]
            base = best_x[i]
            # Adaptive step: start with a reasonable range
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
    current_area = _compute_area_coarse(x, n)
    best_x = x.copy()
    best_area = current_area
    step_size = 0.01

    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Pick a random coordinate
        i = np.random.randint(n)
        # Try several perturbations
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
            # Occasionally shrink step size if we've been improving steadily
            if it % 50 == 49:
                step_size *= 0.8
        else:
            # No improvement; shrink step size slightly
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
    deadline = start_time + 620  # Safety margin: 620s out of 630s

    # Initialize with Keich construction
    x = _keich_init(n)
    current_area = _compute_area_coarse(x, n)
    best_x = x.copy()
    best_area = current_area

    # Phase 1: Coordinate descent on coarse grid
    if time.time() < deadline - 60:
        x, area = _coordinate_descent(x, n, max_iters=100, deadline=deadline - 60)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 2: Local search with random restarts
    if time.time() < deadline - 30:
        x, area = _local_search(best_x, n, max_iters=200, deadline=deadline - 30)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 3: More coordinate descent passes
    if time.time() < deadline - 15:
        x, area = _coordinate_descent(best_x, n, max_iters=100, deadline=deadline - 15)
        if area < best_area:
            best_x, best_area = x, area

    # Phase 4: Final local search polish
    if time.time() < deadline - 5:
        x, area = _local_search(best_x, n, max_iters=200, deadline=deadline - 5)
        if area < best_area:
            best_x, best_area = x, area

    # Verify with fine grid
    fine_area = _compute_area_fine(best_x, n)
    if not np.isfinite(fine_area) or fine_area <= 0:
        # Fallback to Keich
        return _keich_init(n)

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
