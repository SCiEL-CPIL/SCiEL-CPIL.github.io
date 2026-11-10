# EVOLVE-BLOCK-START
import time
import numpy as np

DEADLINE = 630.0
START_TIME = time.time()


def _remaining() -> float:
    return DEADLINE - (time.time() - START_TIME)


def _fast_volume(x, y, N=8, sample=200000):
    """
    Deterministic, fast volume estimator using a fixed grid of c-values and
    vectorized interval-union in x and y, then a 2D coverage count via 1D
    projection trick.

    Since each tube at height c is a product of an x-interval (depending on i)
    and a y-interval (depending on j), the union over tubes is NOT a simple
    product in general (because different tubes have different i,j). However,
    we can still estimate volume by sampling c and for each c computing the
    union area in the xy-plane, then averaging over c.

    For a fixed c, tube (i,j) covers:
        x in [x_k + c*i/N, x_k + c*i/N + 1/N]
        y in [y_k + c*j/N, y_k + c*j/N + 1/N]

    The union of 64 axis-aligned rectangles in 2D. We can compute the union
    area exactly via sweep-line, but for speed we use a fine grid.

    Actually, a faster approach: for each c, the x-projection of the union is
    the union of intervals [x_k + c*i/N, x_k + c*i/N + 1/N] for all k, and
    similarly for y. But the union area is NOT the product of the 1D union
    lengths because the rectangles are correlated (each rectangle pairs a
    specific x-interval with a specific y-interval).

    Let's use a moderate grid for the 2D union area at each c.
    """
    # Use a set of c-values
    n_c = 20
    c_vals = np.linspace(0.05, 0.95, n_c)

    total_area = 0.0
    for c in c_vals:
        # For each tube k, compute x and y intervals
        i = np.arange(N)[:, None] * np.ones((N, N))  # shape (N, N)
        j = np.ones((N, N)) * np.arange(N)[None, :]  # shape (N, N)

        x_k = x.reshape(N, N)
        y_k = y.reshape(N, N)

        x_lo = x_k + c * i / N
        x_hi = x_lo + 1.0 / N
        y_lo = y_k + c * j / N
        y_hi = y_lo + 1.0 / N

        # Compute union area of 64 rectangles via grid
        x_min = x_lo.min()
        x_max = x_hi.max()
        y_min = y_lo.min()
        y_max = y_hi.max()

        # Use a 64x64 grid for the union area
        n_grid = 64
        x_centers = np.linspace(x_min, x_max, n_grid + 1)
        y_centers = np.linspace(y_min, y_max, n_grid + 1)
        dx = (x_max - x_min) / n_grid
        dy = (y_max - y_min) / n_grid

        # Create grid points
        xg, yg = np.meshgrid(x_centers[:-1] + dx / 2, y_centers[:-1] + dy / 2, indexing="ij")

        # Check if each grid point is inside any rectangle
        # For speed, use broadcasting with all 64 tubes
        # xg shape: (n_grid, n_grid), x_lo shape: (N, N)
        # Reshape to (n_grid, n_grid, 1) and (1, 1, N, N)
        inside_x = (xg[..., None, None] >= x_lo[None, None]) & (xg[..., None, None] <= x_hi[None, None])
        inside_y = (yg[..., None, None] >= y_lo[None, None]) & (yg[..., None, None] <= y_hi[None, None])
        inside = inside_x & inside_y  # shape (n_grid, n_grid, N, N)

        covered = inside.any(axis=(-2, -1))  # shape (n_grid, n_grid)
        area = covered.sum() * dx * dy
        total_area += area

    return total_area / n_c


def _fast_volume_mc(x, y, N=8, sample=100000, seed=42):
    """
    Monte Carlo volume estimator using a fixed seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    x_k = x
    y_k = y

    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)

    X0 = x_k.min()
    X1 = (x_k + i_flat / N + 1.0 / N).max()
    Y0 = y_k.min()
    Y1 = (y_k + j_flat / N + 1.0 / N).max()

    if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
        return 1e9

    a = rng.uniform(X0, X1, sample)
    b = rng.uniform(Y0, Y1, sample)
    c = rng.uniform(0, 1, sample)

    # Check if each point is inside any tube
    # For each sample point, check all 64 tubes
    # a, b, c shape: (sample,)
    # x_k, i_flat shape: (N*N,)
    # tube k covers x in [x_k[k] + c* i_flat[k]/N, x_k[k] + c*i_flat[k]/N + 1/N]
    # and y in [y_k[k] + c*j_flat[k]/N, y_k[k] + c*j_flat[k]/N + 1/N]

    # Broadcasting: a[:, None] vs x_k[None, :]
    x_lo = x_k[None, :] + c[:, None] * i_flat[None, :] / N
    x_hi = x_lo + 1.0 / N
    y_lo = y_k[None, :] + c[:, None] * j_flat[None, :] / N
    y_hi = y_lo + 1.0 / N

    inside_x = (a[:, None] >= x_lo) & (a[:, None] <= x_hi)
    inside_y = (b[:, None] >= y_lo) & (b[:, None] <= y_hi)
    inside = inside_x & inside_y

    covered = inside.any(axis=1)
    fraction = covered.mean()
    return fraction * (X1 - X0) * (Y1 - Y0)


def _keich_init(N=8):
    """Generate the Keich binary construction."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


def _evaluate(positions, N=8):
    """Evaluate the volume for given positions."""
    x = positions[:, 0]
    y = positions[:, 1]
    return _fast_volume_mc(x, y, N=N, sample=100000, seed=42)


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = 630.0

    # Start with Keich initialization
    best_pos = _keich_init(N)
    best_vol = _evaluate(best_pos, N)

    # Local search using gradient-free optimization
    # Use a simple random perturbation approach with adaptive step size
    rng = np.random.default_rng(123)

    step_size = 0.01
    n_iter = 0
    max_iter = 5000

    while n_iter < max_iter:
        if time.time() > deadline - 60:  # safety margin
            break

        # Generate perturbation
        perturbation = rng.normal(0, step_size, size=best_pos.shape)
        candidate = best_pos + perturbation

        vol = _evaluate(candidate, N)

        if vol < best_vol:
            best_pos = candidate
            best_vol = vol
            step_size = max(step_size * 0.95, 0.001)
        else:
            step_size = min(step_size * 1.05, 0.1)

        n_iter += 1

    # Final high-precision evaluation with the exact MC method from the task
    # to ensure correctness
    x = best_pos[:, 0]
    y = best_pos[:, 1]

    i_flat = np.repeat(np.arange(N), N)
    j_flat = np.tile(np.arange(N), N)

    X0 = x.min()
    X1 = (x + i_flat / N + 1.0 / N).max()
    Y0 = y.min()
    Y1 = (y + j_flat / N + 1.0 / N).max()

    if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
        # Return Keich if invalid
        return _keich_init(N)

    rng_final = np.random.default_rng(12345)
    a = rng_final.uniform(X0, X1, 500000)
    b = rng_final.uniform(Y0, Y1, 500000)
    c = rng_final.uniform(0, 1, 500000)

    x_lo = x[None, :] + c[:, None] * i_flat[None, :] / N
    x_hi = x_lo + 1.0 / N
    y_lo = y[None, :] + c[:, None] * j_flat[None, :] / N
    y_hi = y_lo + 1.0 / N

    inside_x = (a[:, None] >= x_lo) & (a[:, None] <= x_hi)
    inside_y = (b[:, None] >= y_lo) & (b[:, None] <= y_hi)
    inside = inside_x & inside_y
    covered = inside.any(axis=1)
    V = covered.mean() * (X1 - X0) * (Y1 - Y0)

    if not np.isfinite(V) or V <= 0:
        return _keich_init(N)

    return best_pos


# EVOLVE-BLOCK-END
