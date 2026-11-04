"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# Deadline guard: 140s wall-clock (10s safety margin under 150s)
_DEADLINE = 140.0
_START = None


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _soft_objective(x, n=N, mu=1000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    All computations are vectorized.
    """
    n = len(x) // 3
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(rs)

    # Boundary violations
    b_left = np.maximum(0, rs - xs)
    b_bot = np.maximum(0, rs - ys)
    b_right = np.maximum(0, rs - (1.0 - xs))
    b_top = np.maximum(0, rs - (1.0 - ys))
    bv = b_left + b_bot + b_right + b_top
    bpen = np.sum(bv ** 2)

    # Overlap violations (vectorized)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)
    rs_sum = rs[i_idx] + rs[j_idx]
    ov = np.maximum(0, rs_sum - dist)
    open_ = np.sum(ov ** 2)

    total_penalty = bpen + open_
    energy = obj + mu * total_penalty

    # Gradient (vectorized)
    grad = np.zeros_like(x)

    # d(-sum r)/dr_i = -1
    grad[2::3] = -1.0

    # Boundary penalty gradient
    # d(bpen)/dxi = 2*bv * (db_left/dxi + db_right/dxi)
    # db_left/dxi = -1 if b_left > 0 else 0
    # db_right/dxi = +1 if b_right > 0 else 0
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    # Overlap penalty gradient (vectorized)
    # For each pair (i,j) with ov > 0:
    # d(ov)/dxi_i = -(xi-xj)/dist, d(ov)/dxi_j = +(xi-xj)/dist
    # d(ov)/dri = 1, d(ov)/drj = 1
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        # Handle degenerate case
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

        # Accumulate gradients using np.add.at
        np.add.at(grad, 3 * i_m, -mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * i_m + 1, -mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * i_m + 2, mu * 2 * ov_m)
        np.add.at(grad, 3 * j_m, mu * 2 * ov_m * ux)
        np.add.at(grad, 3 * j_m + 1, mu * 2 * ov_m * uy)
        np.add.at(grad, 3 * j_m + 2, mu * 2 * ov_m)

    return energy, grad


def _is_valid(x, n=N, tol=1e-6):
    """Check if configuration is valid (all circles inside, no overlaps)."""
    pts = x.reshape(n, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    # Boundary
    if np.any(xs - rs < -tol) or np.any(ys - rs < -tol):
        return False
    if np.any(1.0 - xs - rs < -tol) or np.any(1.0 - ys - rs < -tol):
        return False
    # Overlap
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if np.any(rs[i_idx] + rs[j_idx] - dist > tol):
        return False
    return True


def _project_feasible(x, n=N, max_iter=500, tol=1e-10):
    """
    Project a configuration to feasibility by iteratively resolving violations.
    Reduces radii and adjusts positions to eliminate boundary/overlap violations.
    Vectorized overlap resolution.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(max_iter):
        xs, ys, rs = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
        
        # Fix boundary violations: reduce radius to fit
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        min_d = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top))
        too_big = rs > min_d
        if np.any(too_big):
            rs[too_big] = min_d[too_big] * (1 - 1e-9)
        
        # Fix overlap violations: vectorized radius reduction
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)
        
        if not np.any(ov > tol):
            pts[:, 2] = rs
            # Final boundary check
            xs_f, ys_f, rs_f = pts[:, 0], pts[:, 1], pts[:, 2]
            d_l = xs_f; d_b = ys_f; d_r = 1.0 - xs_f; d_t = 1.0 - ys_f
            min_df = np.minimum(np.minimum(d_l, d_b), np.minimum(d_r, d_t))
            if not np.any(rs_f > mindf + tol):
                return pts.ravel()
            # Still have boundary issues, continue
            pts[:, 2] = rs
            continue
        
        # Vectorized: accumulate required reductions per circle
        reductions = np.zeros(n)
        np.add.at(reductions, i_idx, ov / 2 + tol)
        np.add.at(reductions, j_idx, ov / 2 + tol)
        rs = np.maximum(0.01, rs - reductions)
        pts[:, 2] = rs
    
    return pts.ravel()


def _score(x, n=N):
    """Return sum of radii if valid, else -inf."""
    pts = x.reshape(n, 3)
    if not _is_valid(pts.ravel(), n):
        return -np.inf
    return np.sum(pts[:, 2])


def _grow_radii(x, n=N, max_iter=100, tol=1e-12):
    """
    Try to increase radii while maintaining feasibility.
    For each circle, find the maximum radius increase allowed by
    boundary and overlap constraints, then take a fraction of it.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(max_iter):
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # Max increase allowed by boundary for each circle
        d_left = xs
        d_bot = ys
        d_right = 1.0 - xs
        d_top = 1.0 - ys
        max_boundary = np.minimum(np.minimum(d_left, d_bot), np.minimum(d_right, d_top)) - rs
        
        # Max increase allowed by overlaps
        # For pair (i,j): rs[i] + rs[j] + delta_i + delta_j <= dist
        # If we increase both by same amount delta: 2*delta <= dist - rs[i] - rs[j]
        # So delta <= (dist - rs[i] - rs[j]) / 2
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        slack = dist - rs[i_idx] - rs[j_idx]
        max_overlap = np.full(n, np.inf)
        np.minimum.at(max_overlap, i_idx, slack / 2)
        np.minimum.at(max_overlap, j_idx, slack / 2)
        
        # Combined max increase per circle
        max_inc = np.minimum(max_boundary, max_overlap)
        max_inc = np.maximum(0, max_inc)
        
        if np.all(max_inc < tol):
            break
        
        # Take a conservative fraction
        step = 0.5 * max_inc
        rs_new = rs + step
        pts[:, 2] = rs_new
    
    return pts.ravel()


def _hex_packing_init(n=N):
    """Hexagonal packing initial guess."""
    # For 26 circles, use a hexagonal-like arrangement
    row_counts = [5, 5, 5, 5, 4, 2]
    r_est = 0.09
    y_spacing = r_est * np.sqrt(3)
    x_spacing = 2 * r_est
    
    xs, ys, rs = [], [], []
    for i, count in enumerate(row_counts):
        y = (i + 0.5) * y_spacing
        total_width = (count - 1) * x_spacing
        start_x = 0.5 - total_width / 2
        for j in range(count):
            x = start_x + j * x_spacing
            if i % 2 == 1:
                x += x_spacing / 2
            xs.append(x)
            ys.append(y)
            rs.append(r_est)
    
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    rs_arr = np.array(rs)
    
    min_x = np.min(xs_arr - rs_arr)
    max_x = np.max(xs_arr + rs_arr)
    min_y = np.min(ys_arr - rs_arr)
    max_y = np.max(ys_arr + rs_arr)
    
    width = max_x - min_x
    height = max_y - min_y
    scale = min(1.0 / width, 1.0 / height)
    
    xs_arr = (xs_arr - min_x) * scale
    ys_arr = (ys_arr - min_y) * scale
    rs_arr = rs_arr * scale
    
    xs_arr += (1.0 - width * scale) / 2
    ys_arr += (1.0 - height * scale) / 2
    
    x = np.zeros(3 * n)
    x[0::3] = xs_arr
    x[1::3] = ys_arr
    x[2::3] = rs_arr
    
    return x


def _random_init(n=N, seed=None):
    """Random initial guess."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0.1, 0.9, n)
    ys = rng.uniform(0.1, 0.9, n)
    rs = rng.uniform(0.05, 0.12, n)
    x = np.zeros(3 * n)
    x[0::3] = xs
    x[1::3] = ys
    x[2::3] = rs
    return x


def _grid_init(n=N):
    """Grid initial guess."""
    side = int(np.ceil(np.sqrt(n)))
    xs, ys, rs = [], [], []
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            xs.append(x)
            ys.append(y)
            rs.append(r)
            count += 1
    x = np.zeros(3 * n)
    x[0::3] = xs[:n]
    x[1::3] = ys[:n]
    x[2::3] = rs[:n]
    return x


def construct():
    global _START
    _START = time.time()
    
    best_x = None
    best_score = -np.inf
    
    # Try multiple initializations
    inits = []
    
    # Hex packing
    inits.append(_hex_packing_init(N))
    
    # Random inits with varied seeds
    for seed in [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 7777, 8888, 9999]:
        inits.append(_random_init(N, seed=seed))
    
    # Grid init
    inits.append(_grid_init(N))
    
    # Additional structured inits: two-row, three-row configurations
    # Two rows of 13
    n_per_row = 13
    r_est = 1.0 / (2 * n_per_row + 1)
    xs2, ys2, rs2 = [], [], []
    for row in range(2):
        y = (row + 0.5) / 2.0
        for j in range(n_per_row):
            x = (j + 0.5) / n_per_row
            xs2.append(x)
            ys2.append(y)
            rs2.append(r_est)
    x2 = np.zeros(3 * N)
    x2[0::3] = xs2[:N]
    x2[1::3] = ys2[:N]
    x2[2::3] = rs2[:N]
    inits.append(x2)
    
    # Three rows: 9, 9, 8
    row_counts3 = [9, 9, 8]
    r_est3 = 0.1
    xs3, ys3, rs3 = [], [], []
    for i, count in enumerate(row_counts3):
        y = (i + 0.5) / 3.0
        total_width = (count - 1) * 2 * r_est3
        start_x = 0.5 - total_width / 2
        for j in range(count):
            x = start_x + j * 2 * r_est3
            if i % 2 == 1:
                x += r_est3
            xs3.append(x)
            ys3.append(y)
            rs3.append(r_est3)
    x3 = np.zeros(3 * N)
    x3[0::3] = xs3[:N]
    x3[1::3] = ys3[:N]
    x3[2::3] = rs3[:N]
    inits.append(x3)
    
    for init_x in inits:
        if _remaining() < 12.0:
            break
        
        # Optimize
        try:
            res = minimize(
                _soft_objective,
                init_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'ftol': 1e-14, 'gtol': 1e-9, 'maxls': 50}
            )
            x_opt = res.x
            
            # Project to feasibility if needed
            if not _is_valid(x_opt, N, tol=1e-8):
                x_proj = _project_feasible(x_opt, N)
                if _is_valid(x_proj, N, tol=1e-8):
                    x_opt = x_proj
            
            score = _score(x_opt, N)
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Polish: if we have a valid solution, do a final high-precision pass
    if best_x is not None and _remaining() > 8.0:
        try:
            # Use a smaller mu for polish since we're already near feasible
            res = minimize(
                lambda x: _soft_objective(x, mu=100.0),
                best_x,
                method='L-BFGS-B',
                jac=lambda x: _soft_objective(x, mu=100.0)[1],
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 100}
            )
            x_opt = res.x
            
            # Project to feasibility if needed
            if not _is_valid(x_opt, N, tol=1e-8):
                x_proj = _project_feasible(x_opt, N)
                if _is_valid(x_proj, N, tol=1e-8):
                    x_opt = x_proj
            
            # Try to grow radii
            if _is_valid(x_opt, N, tol=1e-8) and _remaining() > 4.0:
                x_grown = _grow_radii(x_opt, N)
                if _is_valid(x_grown, N, tol=1e-8):
                    score_grown = _score(x_grown, N)
                    if score_grown > _score(x_opt, N):
                        x_opt = x_grown
            
            score = _score(x_opt, N)
            if score > best_score:
                best_score = score
                best_x = x_opt.copy()
        except Exception:
            pass
    
    # If still no valid solution, fall back to grid
    if best_x is None or not _is_valid(best_x, N):
        best_x = _grid_init(N)
    
    return best_x.reshape(N, 3)
