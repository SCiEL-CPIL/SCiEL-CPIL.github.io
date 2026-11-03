import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 110.0  # Safety margin under 150s kill


def _elapsed(start):
    return time.time() - start


def _penalized(params, N, w_overlap=50.0, w_bound=50.0):
    """
    Vectorized penalty-based objective.
    Minimize: -sum(r) + w_bound * boundary_penalty + w_overlap * overlap_penalty
    params: shape (N, 3) = [x, y, r]
    """
    x = params[:, 0]
    y = params[:, 1]
    r = params[:, 2]

    obj = -np.sum(r)

    # Boundary penalty: r must be <= min(x, 1-x, y, 1-y)
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    over = np.maximum(r - d, 0.0)
    pen_bound = w_bound * np.sum(over ** 2)

    # Overlap penalty: dist(i,j) >= r_i + r_j
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    viol = np.maximum(need - dist, 0.0)
    # Zero out diagonal (self-distance is 0, need is 2r, but we don't want self-penalty)
    np.fill_diagonal(viol, 0.0)
    pen_overlap = w_overlap * np.sum(viol ** 2)

    return obj + pen_bound + pen_overlap


def _valid(x, y, r, eps=1e-7):
    """Check if packing is valid (no boundary violations, no overlaps)."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    if np.any(r + eps > d):
        return False
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    need = r[:, None] + r[None, :]
    np.fill_diagonal(need, 0.0)
    np.fill_diagonal(dist, np.inf)
    if np.any(need + eps > dist):
        return False
    return True


def _fix_to_boundary(x, y, r):
    """Clip radii so each circle fits inside the square."""
    d = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    return np.minimum(r, np.maximum(d, 1e-6))


def _fix_overlaps(x, y, r, max_iter=50):
    """Iteratively shrink radii to resolve overlaps. Returns (x, y, r_fixed)."""
    x = x.copy()
    y = y.copy()
    r = r.copy()
    for _ in range(max_iter):
        changed = False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        need = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(need, 0.0)
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < need[i, j]:
                    # Shrink both proportionally
                    scale = dist[i, j] / need[i, j] * 0.99
                    r[i] *= scale
                    r[j] *= scale
                    changed = True
        if not changed:
            break
    return x, y, r


def _grid_init_5x5_plus1(rng):
    """5x5 grid with one extra circle."""
    cols, rows = 5, 5
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.08), (0.1, 0.5), (0.9, 0.5)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.18)
    return x, y, r


def _grid_init_4x6_plus2(rng):
    """4x6 grid (24 circles) plus 2 extra."""
    cols, rows = 4, 6
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    extra = [(0.5, 0.5), (0.25, 0.75)]
    pts = pts + extra[:N - len(pts)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.17)
    return x, y, r


def _grid_init_3x9(rng):
    """3x9 grid (27 circles) take first 26."""
    cols, rows = 3, 9
    xs = (np.arange(cols) + 0.5) / cols
    ys = (np.arange(rows) + 0.5) / rows
    pts = [(xs[i], ys[j]) for j in range(rows) for i in range(cols)]
    x = np.array([p[0] for p in pts[:N]])
    y = np.array([p[1] for p in pts[:N]])
    r = np.full(N, 0.15)
    return x, y, r


def _random_init(rng):
    """Random positions with small radii."""
    x = rng.uniform(0.05, 0.95, N)
    y = rng.uniform(0.05, 0.95, N)
    r = np.full(N, 0.12)
    return x, y, r


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    # Generate diverse initial layouts
    init_funcs = [
        _grid_init_5x5_plus1,
        _grid_init_4x6_plus2,
        _grid_init_3x9,
        _random_init,
    ]

    best_params = None
    best_score = -np.inf

    # Fallback: simple valid grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    cells = cells[:N]
    while len(cells) < N:
        cells.append((0.5, 0.5, 0.01))
    fallback = np.array(cells[:N])
    best_params = fallback.copy()
    best_score = np.sum(fallback[:, 2])

    for init_func in init_funcs:
        if _elapsed(start_time) > DEADLINE - 10.0:
            break

        x, y, r = init_func(rng)
        r = _fix_to_boundary(x, y, r)

        # Try multiple restarts with jitter
        for restart in range(4):
            if _elapsed(start_time) > DEADLINE - 5.0:
                break

            x0 = x.copy() + rng.normal(0, 0.01, N)
            y0 = y.copy() + rng.normal(0, 0.01, N)
            r0 = r.copy()
            x0 = np.clip(x0, 0.01, 0.99)
            y0 = np.clip(y0, 0.01, 0.99)
            r0 = np.clip(r0, 0.01, 0.45)

            vec0 = np.column_stack([x0, y0, r0])

            # L-BFGS-B with increasing penalty weights
            for w_mult in [1.0, 2.0, 5.0]:
                if _elapsed(start_time) > DEADLINE - 5.0:
                    break

                def objective(params, wm=w_mult):
                    return _penalized(params, N, w_overlap=50.0 * wm, w_bound=50.0 * wm)

                try:
                    res = minimize(
                        objective,
                        vec0,
                        method='L-BFGS-B',
                        bounds=[(0.001, 0.999)] * N + [(0.001, 0.999)] * N + [(0.001, 0.45)] * N,
                        options={'maxiter': 300, 'ftol': 1e-8}
                    )
                    candidate = res.x.reshape(N, 3)
                    candidate[:, 0] = np.clip(candidate[:, 0], 0.001, 0.999)
                    candidate[:, 1] = np.clip(candidate[:, 1], 0.001, 0.999)
                    candidate[:, 2] = np.clip(candidate[:, 2], 0.001, 0.45)

                    # Fix boundary
                    candidate[:, 2] = _fix_to_boundary(candidate[:, 0], candidate[:, 1], candidate[:, 2])

                    # Check validity
                    if _valid(candidate[:, 0], candidate[:, 1], candidate[:, 2]):
                        score = np.sum(candidate[:, 2])
                        if score > best_score:
                            best_score = score
                            best_params = candidate.copy()
                            vec0 = candidate.copy()  # Use best as next start
                except Exception:
                    pass

            # Nelder-Mead as secondary optimizer (robust to non-smoothness)
            if _elapsed(start_time) > DEADLINE - 5.0:
                break

            def objective_nm(params):
                return _penalized(params, N, w_overlap=30.0, w_bound=30.0)

            try:
                res = minimize(
                    objective_nm,
                    vec0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 3000}
                )
                candidate = res.x.reshape(N, 3)
                candidate[:, 0] = np.clip(candidate[:, 0], 0.001, 0.999)
                candidate[:, 1] = np.clip(candidate[:, 1], 0.001, 0.999)
                candidate[:, 2] = np.clip(candidate[:, 2], 0.001, 0.45)
                candidate[:, 2] = _fix_to_boundary(candidate[:, 0], candidate[:, 1], candidate[:, 2])

                if _valid(candidate[:, 0], candidate[:, 1], candidate[:, 2]):
                    score = np.sum(candidate[:, 2])
                    if score > best_score:
                        best_score = score
                        best_params = candidate.copy()
            except Exception:
                pass

    # Final cleanup: ensure strict validity
    final = best_params.copy()
    for _ in range(20):
        changed = False
        # Fix boundaries
        d = np.minimum(np.minimum(final[:, 0], 1 - final[:, 0]), np.minimum(final[:, 1], 1 - final[:, 1]))
        mask = final[:, 2] > d
        if np.any(mask):
            final[mask, 2] = d[mask] * 0.999
            changed = True

        # Fix overlaps
        x = final[:, 0]
        y = final[:, 1]
        r = final[:, 2]
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy)
        need = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(need, 0.0)
        for i in range(N):
            for j in range(i + 1, N):
                if dist[i, j] < need[i, j]:
                    scale = dist[i, j] / need[i, j] * 0.99
                    final[i, 2] *= scale
                    final[j, 2] *= scale
                    changed = True
        if not changed:
            break

    # Ensure final validity
    if not _valid(final[:, 0], final[:, 1], final[:, 2], eps=1e-5):
        # Use fallback
        final = fallback.copy()

    return final
