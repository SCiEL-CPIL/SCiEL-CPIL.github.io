# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DL = 600.0
_MARGIN = 15.0


def _deadline_ok():
    return (time.time() - _T0) < (_DL - _MARGIN)


def _keich(N):
    pos = int(round(np.log2(N)))
    j_terms = np.arange(1, pos + 1, dtype=float)
    out = np.empty(N, dtype=np.float64)
    for idx in range(N):
        bits = [int(b) for b in bin(idx)[2:]]
        bits = [0] * (pos - len(bits)) + bits
        out[idx] = np.sum((1.0 - j_terms) * np.array(bits) * (2.0 ** -j_terms) / pos)
    return out


def _keich_positions(N):
    keich = _keich(N)
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    base_x = keich[i_grid.flatten()]
    base_y = keich[j_grid.flatten()]
    return np.stack([base_x, base_y], axis=1)


def _union_len(p, c, N):
    """Union length of intervals [p_i + c*i/N, p_i + c*i/N + 1/N] at height c."""
    slopes = np.arange(N, dtype=np.float64) / N
    L = p + c * slopes
    R = L + 1.0 / N
    idx = np.argsort(L)
    Ls = L[idx]
    Rs = R[idx]
    # Merge overlapping intervals
    cur_r = Rs[0]
    total = 0.0
    for k in range(1, N):
        if Ls[k] > cur_r:
            total += cur_r - Ls[k - 1]
            cur_r = Rs[k]
        else:
            if Rs[k] > cur_r:
                cur_r = Rs[k]
    total += cur_r - Ls[N - 1]
    return total


def _union_len_vec(p, c_arr, N):
    """Vectorized union length for multiple c values."""
    slopes = np.arange(N, dtype=np.float64) / N
    L = p[None, :] + c_arr[:, None] * slopes[None, :]
    R = L + 1.0 / N
    idx = np.argsort(L, axis=1)
    Ls = np.take_along_axis(L, idx, axis=1)
    Rs = np.take_along_axis(R, idx, axis=1)
    # Merge: for each row, compute union length
    cur_r = Rs[:, 0].copy()
    total = np.zeros(len(c_arr), dtype=np.float64)
    for k in range(1, N):
        gap = Ls[:, k] > cur_r
        total += np.where(gap, cur_r - Ls[:, k - 1], 0.0)
        # For non-gap: cur_r = max(cur_r, Rs[:,k])
        # For gap: cur_r = Rs[:,k]
        cur_r = np.where(gap, Rs[:, k], np.maximum(cur_r, Rs[:, k]))
    total += cur_r - Ls[:, N - 1]
    return total


def _product_integral(p_x, p_y, N, n_c=256):
    """Compute ∫ Ux(c)*Uy(c) dc using trapezoidal rule."""
    c = np.linspace(0.0, 1.0, n_c)
    ux = _union_len_vec(p_x, c, N)
    uy = _union_len_vec(p_y, c, N)
    prod = ux * uy
    return np.trapz(prod, c)


def _box_area_x(p_x, N):
    """X1 - X0 where X0=min(p_x), X1=max(p_x + (i+1)/N)."""
    slopes = np.arange(N, dtype=np.float64) / N
    x0 = np.min(p_x)
    x1 = np.max(p_x + slopes)
    return x1 - x0


def _box_area_y(p_y, N):
    """Y1 - Y0 where Y0=min(p_y), Y1=max(p_y + (j+1)/N)."""
    slopes = np.arange(N, dtype=np.float64) / N
    y0 = np.min(p_y)
    y1 = np.max(p_y + slopes)
    return y1 - y0


def _mc_volume(positions, N, seed=12345):
    """Compute the actual MC volume as defined in the task."""
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    idx = np.arange(N * N)
    i_vals = idx // N
    j_vals = idx % N
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i_vals + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j_vals + 1) / N)
    box_area = (X1 - X0) * (Y1 - Y0)
    if box_area <= 1e-9:
        return 0.0
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0, 1, 500000)
    # Check if any tube contains (a, b, c)
    # Tube k: x in [x_k + c*i/8, x_k + c*i/8 + 1/8], y in [y_k + c*j/8, y_k + c*j/8 + 1/8]
    inside = np.zeros(500000, dtype=bool)
    for k in range(N * N):
        i_val = i_vals[k]
        j_val = j_vals[k]
        x_lo = x_k[k] + c * i_val / N
        x_hi = x_lo + 1.0 / N
        y_lo = y_k[k] + c * j_val / N
        y_hi = y_lo + 1.0 / N
        inside |= (a >= x_lo) & (a <= x_hi) & (b >= y_lo) & (b <= y_hi)
    frac = np.mean(inside)
    return frac * box_area


def _objective_product(p_combined, N, n_c=128):
    """Objective: product integral * box areas (proportional to volume)."""
    p_x = p_combined[:N]
    p_y = p_combined[N:]
    px = _box_area_x(p_x, N)
    py = _box_area_y(p_y, N)
    if px <= 1e-9 or py <= 1e-9:
        return 1e6
    pi = _product_integral(p_x, p_y, N, n_c)
    return px * py * pi


def _make_init(N, seed):
    """Create initial positions using keich with perturbation."""
    keich = _keich(N)
    rng = np.random.default_rng(seed)
    p_x = keich + rng.uniform(-0.05, 0.05, N)
    p_y = keich + rng.uniform(-0.05, 0.05, N)
    return np.concatenate([p_x, p_y])


def _bounds_1d(N):
    """Bounds for each 1D position: must keep intervals within reasonable range."""
    # Each p_i should be in [-1, 2] roughly to keep box finite
    lo = np.full(N, -0.5, dtype=np.float64)
    hi = np.full(N, 1.5, dtype=np.float64)
    return list(zip(lo, hi))


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    try:
        keich = _keich(N)
        best_positions = _keich_positions(N)
        best_score = 0.0  # we want to minimize volume, so track 1/V

        # Multi-start optimization of the joint (p_x, p_y) product integral
        seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        for seed in seeds:
            if not _deadline_ok():
                break
            try:
                p0 = _make_init(N, seed)
                bounds = _bounds_1d(N) + _bounds_1d(N)
                res = minimize(_objective_product, p0, args=(N, 128),
                               method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 200, "ftol": 1e-12,
                                        "gtol": 1e-10, "maxls": 30})
                p_opt = res.x
                p_x = p_opt[:N]
                p_y = p_opt[N:]

                # Polish with higher resolution
                if _deadline_ok():
                    res2 = minimize(_objective_product, p_opt, args=(N, 256),
                                    method="L-BFGS-B", bounds=bounds,
                                    options={"maxiter": 100, "ftol": 1e-13,
                                             "gtol": 1e-12, "maxls": 30})
                    if res2.success:
                        p_opt = res2.x
                        p_x = p_opt[:N]
                        p_y = p_opt[N:]

                # Build positions
                i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
                base_x = p_x[i_grid.flatten()]
                base_y = p_y[j_grid.flatten()]
                positions = np.stack([base_x, base_y], axis=1)

                if not (np.all(np.isfinite(positions)) and positions.shape == (N * N, 2)):
                    continue

                # Evaluate actual MC volume
                V = _mc_volume(positions, N, seed=12345)
                if V > 0 and np.isfinite(V):
                    score = 1.0 / V
                    if score > best_score:
                        best_score = score
                        best_positions = positions.copy()
            except Exception:
                pass

        # Final fallback: if optimization didn't improve, try asymmetric keich variants
        if best_score <= 0 and _deadline_ok():
            # Try shifting y positions relative to x
            for shift_seed in [10, 11, 12]:
                if not _deadline_ok():
                    break
                rng = np.random.default_rng(shift_seed)
                p_x = keich + rng.uniform(-0.03, 0.03, N)
                p_y = keich + rng.uniform(-0.03, 0.03, N)
                i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
                base_x = p_x[i_grid.flatten()]
                base_y = p_y[j_grid.flatten()]
                positions = np.stack([base_x, base_y], axis=1)
                V = _mc_volume(positions, N, seed=12345)
                if V > 0 and np.isfinite(V):
                    score = 1.0 / V
                    if score > best_score:
                        best_score = score
                        best_positions = positions.copy()

        if best_score <= 0:
            best_positions = _keich_positions(N)

        return best_positions
    except Exception:
        return _keich_positions(8)


# EVOLVE-BLOCK-END
