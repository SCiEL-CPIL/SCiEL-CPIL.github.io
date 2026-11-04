import numpy as np
import time
from scipy.optimize import minimize

N = 26

# Deadline guard: 135s wall-clock (15s safety margin under 150s)
_START = None
_DEADLINE = 135.0


def _elapsed():
    return time.time() - _START


def _remaining():
    return max(0.0, _DEADLINE - _elapsed())


def _energy_and_grad(x, n=N, mu=1000.0):
    """
    x: (3n,) vector: [x0,y0,r0, x1,y1,r1, ...]
    Returns (energy, grad).
    Energy = -sum(r) + mu * (boundary_penalty + overlap_penalty).
    All computations are vectorized.
    """
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
    db_dxi = np.where(b_left > 0, -1.0, 0.0) + np.where(b_right > 0, 1.0, 0.0)
    db_dyi = np.where(b_bot > 0, -1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0)
    db_dri = (np.where(b_left > 0, 1.0, 0.0) + np.where(b_bot > 0, 1.0, 0.0) +
              np.where(b_right > 0, 1.0, 0.0) + np.where(b_top > 0, 1.0, 0.0))

    grad[0::3] += mu * 2 * bv * db_dxi
    grad[1::3] += mu * 2 * bv * db_dyi
    grad[2::3] += mu * 2 * bv * db_dri

    # Overlap penalty gradient (vectorized)
    mask = ov > 0
    if np.any(mask):
        i_m = i_idx[mask]
        j_m = j_idx[mask]
        ov_m = ov[mask]
        dist_m = dist[mask]
        safe_dist = np.where(dist_m < 1e-12, 1.0, dist_m)
        ux = dx[mask] / safe_dist
        uy = dy[mask] / safe_dist

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
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(max_iter):
        if _remaining() < 1.0:
            break
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
            pts[:, 2] = rs
        
        # Fix overlap violations: reduce radii
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        rs_sum = rs[i_idx] + rs[j_idx]
        ov = np.maximum(0, rs_sum - dist)
        
        if not np.any(ov > tol):
            break
        
        # Reduce radii proportionally to resolve overlaps (vectorized)
        overlap_per_circle = np.zeros(n)
        np.add.at(overlap_per_circle, i_idx, ov)
        np.add.at(overlap_per_circle, j_idx, ov)
        
        # Reduce each radius by half its total overlap
        reduction = overlap_per_circle / 2.0 + tol
        rs = np.maximum(0.01, rs - reduction)
        pts[:, 2] = rs
    
    return pts.ravel()


def _grow_radii(x, n=N, max_iter=100, step=0.0005, tol=1e-8):
    """
    Try to grow radii while maintaining feasibility.
    Uses a greedy approach: try to increase each radius slightly,
    check feasibility, and keep the increase if valid.
    """
    pts = x.reshape(n, 3).copy()
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    for _ in range(max_iter):
        if _remaining() < 2.0:
            break
        
        improved = False
        xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # For each circle, compute max possible radius increase
        # limited by boundary and overlaps
        for i in range(n):
            # Boundary limit
            d_left = xs[i]
            d_bot = ys[i]
            d_right = 1.0 - xs[i]
            d_top = 1.0 - ys[i]
            max_r_boundary = min(d_left, d_bot, d_right, d_top)
            
            # Overlap limits with other circles
            max_r_overlap = max_r_boundary
            for j in range(n):
                if i == j:
                    continue
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                # r_i + r_j <= dist => r_i <= dist - r_j
                limit = dist - rs[j]
                if limit < max_r_overlap:
                    max_r_overlap = limit
            
            # Try to grow
            new_r = min(max_r_overlap, rs[i] + step)
            if new_r > rs[i] + 1e-12:
                old_r = rs[i]
                rs[i] = new_r
                pts[i, 2] = new_r
                
                # Check if still valid
                if _is_valid(pts.ravel(), n, tol=tol):
                    improved = True
                else:
                    rs[i] = old_r
                    pts[i, 2] = old_r
        
        if not improved:
            break
    
    return pts.ravel()


def _score(x, n=N):
    """Return sum of radii if valid, else -inf."""
    pts = x.reshape(n, 3)
    if not _is_valid(pts.ravel(), n):
        return -np.inf
    return np.sum(pts[:, 2])


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
        x_start = (1.0 - total_width) / 2.0
        for j in range(count):
            x = x_start + j * x_spacing
            xs.append(x)
            ys.append(y)
            rs.append(r_est)
    
    # If we don't have enough, add random
    while len(xs) < n:
        xs.append(np.random.rand())
        ys.append(np.random.rand())
        rs.append(0.05)
    
    return np.array(xs[:n]), np.array(ys[:n]), np.array(rs[:n])


def _grid_init(n=N):
    """Create initial grid packing."""
    side = int(np.ceil(np.sqrt(n)))
    xs, ys, rs = [], [], []
    for i in range(side):
        for j in range(side):
            if len(xs) >= n:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.005
            xs.append(x)
            ys.append(y)
            rs.append(r)
    
    return np.array(xs), np.array(ys), np.array(rs)


def _random_init(seed, n=N):
    """Create random initial packing."""
    np.random.seed(seed)
    xs = np.random.rand(n)
    ys = np.random.rand(n)
    rs = np.full(n, 0.05)
    return xs, ys, rs


def _best_fit_local_search(centers, radii, max_iters=300, deadline=None):
    """Iteratively move each circle to maximize its radius given others fixed."""
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            best_pos = (centers[i, 0], centers[i, 1])
            best_r = radii[i]
            
            # Coarse: random perturbations
            for _ in range(10):
                angle = np.random.uniform(0, 2 * np.pi)
                dist = np.random.uniform(0, 0.06)
                nx = centers[i, 0] + dist * np.cos(angle)
                ny = centers[i, 1] + dist * np.sin(angle)
                if 0 <= nx <= 1 and 0 <= ny <= 1:
                    max_r = min(nx, 1 - nx, ny, 1 - ny)
                    for j in range(N):
                        if j == i:
                            continue
                        dx = nx - centers[j, 0]
                        dy = ny - centers[j, 1]
                        d = np.sqrt(dx * dx + dy * dy)
                        if d > 0:
                            max_r = min(max_r, d - radii[j])
                    max_r = max(0.0, max_r)
                    
                    if max_r > best_r + 1e-10:
                        best_r = max_r
                        best_pos = (nx, ny)
                        improved = True
            
            # Fine: grid refinement around best position found
            if improved:
                fine_best_r = best_r
                fine_best_pos = best_pos
                for fdx in np.linspace(-0.01, 0.01, 5):
                    for fdy in np.linspace(-0.01, 0.01, 5):
                        fx = best_pos[0] + fdx
                        fy = best_pos[1] + fdy
                        if fx < 0 or fx > 1 or fy < 0 or fy > 1:
                            continue
                        max_r = min(fx, 1 - fx, fy, 1 - fy)
                        for j in range(N):
                            if j == i:
                                continue
                            dx = fx - centers[j, 0]
                            dy = fy - centers[j, 1]
                            d = np.sqrt(dx * dx + dy * dy)
                            if d > 0:
                                max_r = min(max_r, d - radii[j])
                        max_r = max(0.0, max_r)
                        
                        if max_r > fine_best_r + 1e-12:
                            fine_best_r = max_r
                            fine_best_pos = (fx, fy)
                
                best_r = fine_best_r
                best_pos = fine_best_pos
            
            # Update if improved
            if best_r > radii[i] + 1e-10 or best_pos[0] != centers[i, 0] or best_pos[1] != centers[i, 1]:
                centers[i, 0] = best_pos[0]
                centers[i, 1] = best_pos[1]
                radii[i] = best_r
    
    return centers, radii, np.sum(radii)


def _high_precision_polish(centers, radii, deadline=None):
    """
    Final high-precision polish: iteratively adjust each circle to its optimal
    position/radius given others fixed, using a fine grid around current position.
    """
    centers = centers.copy()
    radii = radii.copy()
    
    for iteration in range(30):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            best_pos = (centers[i, 0], centers[i, 1])
            best_r = radii[i]
            
            # Fine grid around current position
            for dx in np.linspace(-0.015, 0.015, 7):
                for dy in np.linspace(-0.015, 0.015, 7):
                    cx = centers[i, 0] + dx
                    cy = centers[i, 1] + dy
                    if cx < 0 or cx > 1 or cy < 0 or cy > 1:
                        continue
                    
                    max_r = min(cx, 1 - cx, cy, 1 - cy)
                    for j in range(N):
                        if j == i:
                            continue
                        ddx = cx - centers[j, 0]
                        ddy = cy - centers[j, 1]
                        dist = np.sqrt(ddx * ddx + ddy * ddy)
                        if dist > 0:
                            max_r = min(max_r, dist - radii[j])
                    max_r = max(0.0, max_r)
                    
                    if max_r > best_r + 1e-12:
                        best_r = max_r
                        best_pos = (cx, cy)
            
            if best_pos[0] != centers[i, 0] or best_pos[1] != centers[i, 1] or best_r > radii[i] + 1e-12:
                centers[i, 0] = best_pos[0]
                centers[i, 1] = best_pos[1]
                radii[i] = best_r
                improved = True
        
        if not improved:
            break
    
    return centers, radii, np.sum(radii)


def _scipy_optimize(x0, max_iters=50, deadline=None):
    """
    Run scipy.optimize.minimize with L-BFGS-B on the energy function.
    Returns optimized x.
    """
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iters, 'gtol': 1e-6, 'disp': False}
        )
        return res.x
    except Exception:
        return x0


def construct():
    global _START
    _START = time.time()
    deadline = _START + 130.0  # Safety margin
    
    best_score = -np.inf
    best_x = None
    
    # Generate multiple initial configurations
    inits = []
    
    # Grid init
    xs, ys, rs = _grid_init()
    x0 = np.column_stack((xs, ys, rs)).ravel()
    inits.append(x0)
    
    # Random inits
    for seed in [42, 123, 456, 789, 1011, 2020, 3030, 4040]:
        xs, ys, rs = _random_init(seed)
        x0 = np.column_stack((xs, ys, rs)).ravel()
        inits.append(x0)
    
    # Hex init
    xs, ys, rs = _hex_packing_init()
    x0 = np.column_stack((xs, ys, rs)).ravel()
    inits.append(x0)
    
    # Process each initialization
    for x0 in inits:
        if _remaining() < 5.0:
            break
        
        try:
            # Step 1: Project to feasible region
            x_feasible = _project_feasible(x0)
            
            # Step 2: Run scipy optimization
            if _remaining() > 10.0:
                x_opt = _scipy_optimize(x_feasible, max_iters=30, deadline=deadline)
            else:
                x_opt = x_feasible
            
            # Step 3: Project again to ensure feasibility
            x_opt = _project_feasible(x_opt)
            
            # Step 4: Grow radii
            x_grown = _grow_radii(x_opt)
            
            # Step 5: Convert to centers/radii for local search
            centers = x_grown.reshape(N, 3)[:, :2]
            radii = x_grown.reshape(N, 3)[:, 2]
            
            # Step 6: Best-fit local search
            if _remaining() > 15.0:
                c_opt, r_opt, score = _best_fit_local_search(centers, radii, 
                                                             max_iters=200, 
                                                             deadline=deadline)
            else:
                c_opt, r_opt, score = centers, radii, np.sum(radii)
            
            # Step 7: High-precision polish
            if _remaining() > 10.0:
                c_hp, r_hp, score_hp = _high_precision_polish(c_opt, r_opt, 
                                                              deadline=deadline)
            else:
                c_hp, r_hp, score_hp = c_opt, r_opt, score
            
            # Step 8: Final grow radii
            x_final = np.column_stack((c_hp, r_hp)).ravel()
            x_final = _grow_radii(x_final)
            
            # Validate
            final_score = _score(x_final)
            if final_score > best_score:
                best_score = final_score
                best_x = x_final.copy()
        except Exception:
            pass
    
    # If no valid solution found, use a simple grid fallback
    if best_x is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([x, y, r])
        return np.array(cells)
    
    return best_x.reshape(N, 3)
