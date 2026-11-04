"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # seconds, safety margin before 150s kill


def _check_validity(circles, W, H):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or y - r < -1e-9 or x + r > W + 1e-9 or y + r > H + 1e-9:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - 1e-9:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=2000, tol=1e-12):
    """
    Directly resolve overlaps and boundary violations using pairwise repulsion.
    Returns (circles, is_valid).
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False

        # Resolve boundary violations by clamping
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        # Resolve overlaps via pairwise separation
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist_sq = dx * dx + dy * dy
                min_dist = r[i] + r[j]
                if dist_sq < (min_dist - tol) ** 2:
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        angle = np.random.uniform(0, 2 * np.pi)
                        dx = np.cos(angle) * 1e-6
                        dy = np.sin(angle) * 1e-6
                        dist = 1e-6
                    overlap = min_dist - dist
                    total_r = r[i] + r[j]
                    wi = r[i] / total_r if total_r > 0 else 0.5
                    wj = r[j] / total_r if total_r > 0 else 0.5
                    push = overlap + 1e-9
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True

        # Re-clamp after separation, shrinking radius if circle doesn't fit
        for i in range(n):
            max_r_x = min(x[i], W - x[i])
            max_r_y = min(y[i], H - y[i])
            max_r = max(1e-4, min(max_r_x, max_r_y))
            if r[i] > max_r:
                r[i] = max_r
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

        if not moved:
            break

    result = np.column_stack([x, y, r])
    valid = _check_validity(result, W, H)
    return result, valid


def _objective_and_grad(params, W, H, eps_scale=1.0):
    """
    Objective: minimize -sum(r) + penalties for violations.
    Returns (obj, grad) with analytical gradient.
    eps_scale controls the smoothness: 1.0 = default, smaller = sharper (more precise)
    """
    n = N
    circles = params.reshape(n, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    eps = 1e-4 * eps_scale
    boundary_weight = 200.0

    def _smooth_max_grad(v):
        smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
        grad_smax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
        return smax, grad_smax

    # x - r >= 0  => violation = r - x
    v = r - x
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y - r >= 0
    v = r - y
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += -2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # x + r <= W
    v = x + r - W
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[0::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # y + r <= H
    v = y + r - H
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[1::3] += 2.0 * boundary_weight * smax * gsmax
    grad[2::3] += 2.0 * boundary_weight * smax * gsmax

    # r >= 0
    v = -r
    smax, gsmax = _smooth_max_grad(v)
    obj += boundary_weight * np.sum(smax ** 2)
    grad[2::3] += -2.0 * boundary_weight * smax * gsmax

    # Overlap penalties: vectorized
    overlap_weight = 200.0
    dx_mat = x[:, np.newaxis] - x[np.newaxis, :]
    dy_mat = y[:, np.newaxis] - y[np.newaxis, :]
    dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
    np.fill_diagonal(dist_mat, 1e12)
    dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)

    iu, ju = np.triu_indices(n, k=1)
    dist_ij = dist_mat_safe[iu, ju]
    dx_ij = dx_mat[iu, ju]
    dy_ij = dy_mat[iu, ju]
    r_sum = r[iu] + r[ju]

    v_vec = r_sum - dist_ij
    smax_vec, gsmax_vec = _smooth_max_grad(v_vec)
    obj += overlap_weight * np.sum(smax_vec ** 2)

    factor_vec = 2.0 * overlap_weight * smax_vec * gsmax_vec
    nx_ij = -dx_ij / dist_ij
    ny_ij = -dy_ij / dist_ij

    grad_x_i = factor_vec * nx_ij
    grad_y_i = factor_vec * ny_ij
    grad_x_j = -factor_vec * nx_ij
    grad_y_j = -factor_vec * ny_ij

    np.add.at(grad, 3 * iu, grad_x_i)
    np.add.at(grad, 3 * iu + 1, grad_y_i)
    np.add.at(grad, 3 * iu + 2, factor_vec)
    np.add.at(grad, 3 * ju, grad_x_j)
    np.add.at(grad, 3 * ju + 1, grad_y_j)
    np.add.at(grad, 3 * ju + 2, factor_vec)

    return obj, grad


def _local_optimize(circles, W, H, max_iter=3000, eps_scale=1.0):
    """Use L-BFGS-B with analytical gradients to locally optimize."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    result = minimize(
        _objective_and_grad, params, args=(W, H, eps_scale),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 200}
    )
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=8000):
    """Four-pass polish with decreasing eps for increasing precision."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    # Pass 1: coarse
    result = minimize(
        _objective_and_grad, params, args=(W, H, 1.0),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 100}
    )
    params = result.x

    # Pass 2: medium
    result = minimize(
        _objective_and_grad, params, args=(W, H, 0.5),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 150}
    )
    params = result.x

    # Pass 3: fine
    result = minimize(
        _objective_and_grad, params, args=(W, H, 0.2),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 200}
    )
    params = result.x

    # Pass 4: ultra-fine
    result = minimize(
        _objective_and_grad, params, args=(W, H, 0.1),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': max_iter // 4, 'ftol': 1e-16, 'gtol': 1e-15, 'maxls': 300}
    )
    return result.x.reshape(N, 3)


def _hexagonal_initialization(W, H):
    """
    Create a hexagonal lattice initial configuration.
    This provides a high-quality starting point close to optimal packing.
    """
    circles = []
    # Hexagonal packing: rows offset by half spacing
    # Try to fit as many circles as possible in a hex pattern
    # Start with a reasonable radius estimate
    # For 21 circles in a rectangle of perimeter 4 (W+H=2)
    # Approximate: if square 1x1, hex spacing ~ 0.22
    
    # Try multiple radius estimates and pick the best fitting one
    best_circles = None
    best_count = 0
    
    for trial_r in np.linspace(0.08, 0.25, 20):
        # Hexagonal spacing
        dx = 2 * trial_r * 1.001  # slight gap
        dy = 2 * trial_r * np.sqrt(3) / 2 * 1.001
        
        row = 0
        y_pos = trial_r + dy / 2
        while y_pos + trial_r <= H + 1e-9 and len(circles) < N:
            offset = 0.0 if row % 2 == 0 else dx / 2
            x_pos = trial_r + offset
            row += 1
            while x_pos + trial_r <= W + 1e-9 and len(circles) < N:
                circles.append([x_pos, y_pos, trial_r])
                x_pos += dx
            y_pos += dy
        
        if len(circles) > best_count:
            best_count = len(circles)
            best_circles = circles.copy()
        
        circles = []
    
    if best_circles is None or len(best_circles) == 0:
        # Fallback: grid
        side = int(np.ceil(np.sqrt(N)))
        for i in range(side):
            for j in range(side):
                if len(circles) >= N:
                    break
                cx = (i + 0.5) * W / side
                cy = (j + 0.5) * H / side
                r = min(W, H) / (2 * side) - 1e-4
                circles.append([cx, cy, r])
    
    # If we have more than N, trim
    if len(circles) > N:
        circles = circles[:N]
    
    # If we have fewer than N, add more with smaller radii
    while len(circles) < N:
        # Find a spot with maximum clearance
        best_x, best_y, best_r = W/2, H/2, 0.01
        for _ in range(50):
            tx = np.random.uniform(0.05, W - 0.05)
            ty = np.random.uniform(0.05, H - 0.05)
            min_dist = min(W, H)
            for c in circles:
                d = np.sqrt((tx - c[0])**2 + (ty - c[1])**2)
                min_dist = min(min_dist, d - c[2])
            min_dist = min(min_dist, tx, W - tx, ty, H - ty)
            if min_dist > best_r:
                best_r = min_dist * 0.9
                best_x, best_y = tx, ty
        best_r = max(0.01, best_r)
        circles.append([best_x, best_y, best_r])
    
    return np.array(circles[:N])


def _grid_initialization(W, H):
    """Create a grid-based initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            cx = (i + 0.5) * W / side
            cy = (j + 0.5) * H / side
            r = min(W, H) / (2 * side) - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:N])


def _random_initialization(W, H, rng):
    """Create a random initial configuration."""
    circles = []
    max_r = min(W, H) / 4
    for _ in range(N):
        r = rng.uniform(0.02, max_r)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _successive_convex_relaxation(circles, W, H, max_sweeps=15, tol=1e-6):
    """
    Successive Convex Relaxation using L-BFGS-B with linearized constraints.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    best_val = -np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    r_max = min(W, H) / 2
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    for sweep in range(max_sweeps):
        def lin_objective_and_grad(params):
            p = params.reshape(n, 3)
            px = p[:, 0]
            py = p[:, 1]
            pr = p[:, 2]

            obj = -np.sum(pr)
            grad = np.zeros_like(params)
            grad[2::3] = -1.0

            eps = 1e-4

            def _smooth_max_grad(v):
                smax = 0.5 * (v + np.sqrt(v ** 2 + eps ** 2))
                gsmax = 0.5 * (1.0 + v / np.sqrt(v ** 2 + eps ** 2))
                return smax, gsmax

            # Boundary penalties
            for v_expr, d_x, d_y, d_r in [
                (pr - px, -1, 0, 1),
                (pr - py, 0, -1, 1),
                (px + pr - W, 1, 0, 1),
                (py + pr - H, 0, 1, 1),
                (-pr, 0, 0, -1),
            ]:
                v = v_expr
                smax, gsmax = _smooth_max_grad(v)
                obj += 100.0 * np.sum(smax ** 2)
                if d_x != 0:
                    grad[0::3] += 200.0 * smax * gsmax * d_x
                if d_y != 0:
                    grad[1::3] += 200.0 * smax * gsmax * d_y
                grad[2::3] += 200.0 * smax * gsmax * d_r

            # Linearized overlap penalties
            for i in range(n):
                for j in range(i + 1, n):
                    dx0 = x[i] - x[j]
                    dy0 = y[i] - y[j]
                    dist0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
                    if dist0 < 1e-10:
                        dist0 = 1e-10
                        dx0 = 1e-10
                        dy0 = 0.0
                    gx_i = dx0 / dist0
                    gy_i = dy0 / dist0

                    lin_dist = dist0 + gx_i * (px[i] - x[i]) + gy_i * (py[i] - y[i]) \
                                 - gx_i * (px[j] - x[j]) - gy_i * (py[j] - y[j])
                    v = (pr[i] + pr[j]) - lin_dist
                    smax, gsmax = _smooth_max_grad(v)
                    obj += 10.0 * smax ** 2

                    d_obj = 20.0 * smax * gsmax
                    grad[3 * i] += d_obj * (-gx_i)
                    grad[3 * i + 1] += d_obj * (-gy_i)
                    grad[3 * i + 2] += d_obj * 1.0
                    grad[3 * j] += d_obj * (gx_i)
                    grad[3 * j + 1] += d_obj * (gy_i)
                    grad[3 * j + 2] += d_obj * 1.0

            return obj, grad

        params = np.column_stack([x, y, r]).flatten()
        result = minimize(
            lin_objective_and_grad, params,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        new_params = result.x.reshape(n, 3)
        new_x = np.clip(new_params[:, 0], new_params[:, 2], W - new_params[:, 2])
        new_y = np.clip(new_params[:, 1], new_params[:, 2], H - new_params[:, 2])
        new_r = np.clip(new_params[:, 2], 0.001, r_max)

        new_val = -np.sum(new_r)
        if new_val < best_val - 1e-8:
            best_val = new_val
            best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()

        if np.max(np.abs(new_x - x)) < tol and np.max(np.abs(new_y - y)) < tol and np.max(np.abs(new_r - r)) < tol:
            break

        x, y, r = new_x, new_y, new_r

    return np.column_stack([best_x, best_y, best_r]), -best_val


def _best_known(W, H, time_budget=30.0):
    """Return the best configuration found so far for given W, H."""
    start = time.time()
    
    # Try hexagonal initialization first (known to be high quality)
    circles_hex = _hexagonal_initialization(W, H)
    best_val = -np.sum(circles_hex[:, 2])
    best_circles = circles_hex.copy()
    
    # Also try grid initialization
    circles_grid = _grid_initialization(W, H)
    if -np.sum(circles_grid[:, 2]) > best_val:
        best_val = -np.sum(circles_grid[:, 2])
        best_circles = circles_grid.copy()

    # Try local optimization with L-BFGS-B on hex init
    circles_opt = _local_optimize(circles_hex, W, H, max_iter=3000, eps_scale=1.0)
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)
    circles_opt, valid_opt = _separate_circles(circles_opt, W, H)
    if valid_opt:
        s = np.sum(circles_opt[:, 2])
        if s > -best_val:
            best_val = -s
            best_circles = circles_opt.copy()

    # Try local optimization on grid init if time permits
    if time.time() - start < time_budget - 10:
        circles_opt2 = _local_optimize(circles_grid, W, H, max_iter=2000, eps_scale=1.0)
        circles_opt2[:, 0] = np.clip(circles_opt2[:, 0], circles_opt2[:, 2], W - circles_opt2[:, 2])
        circles_opt2[:, 1] = np.clip(circles_opt2[:, 1], circles_opt2[:, 2], H - circles_opt2[:, 2])
        circles_opt2[:, 2] = np.clip(circles_opt2[:, 2], 0.001, min(W, H) / 2)
        circles_opt2, valid_opt2 = _separate_circles(circles_opt2, W, H)
        if valid_opt2:
            s = np.sum(circles_opt2[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = circles_opt2.copy()

    # Try successive convex relaxation if time permits
    if time.time() - start < time_budget - 8:
        sc_circles, sc_score = _successive_convex_relaxation(best_circles, W, H, max_sweeps=15)
        sc_circles[:, 0] = np.clip(sc_circles[:, 0], sc_circles[:, 2], W - sc_circles[:, 2])
        sc_circles[:, 1] = np.clip(sc_circles[:, 1], sc_circles[:, 2], H - sc_circles[:, 2])
        sc_circles[:, 2] = np.clip(sc_circles[:, 2], 0.001, min(W, H) / 2)
        sc_circles, valid_sc = _separate_circles(sc_circles, W, H)
        if valid_sc:
            s = np.sum(sc_circles[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = sc_circles.copy()

    # Final polish if time permits
    if time.time() - start < time_budget - 4:
        polished = _polish(best_circles, W, H, max_iter=8000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(W, H) / 2)
        polished, valid_polish = _separate_circles(polished, W, H, max_iter=1000, tol=1e-13)
        if valid_polish:
            s = np.sum(polished[:, 2])
            if s > -best_val:
                best_val = -s
                best_circles = polished.copy()

    return best_circles, -best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Try different aspect ratios (refined around promising region)
    aspect_ratios = [0.88, 0.93, 0.97, 1.0, 1.03, 1.07, 1.12, 1.20, 1.35]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:
            break
        
        # Allocate more time to promising aspect ratios
        if 0.93 <= ar <= 1.07:
            time_budget = 24.0
        elif 0.85 <= ar <= 1.20:
            time_budget = 18.0
        else:
            time_budget = 13.0
        
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        circles, score = _best_known(W, H, time_budget=time_budget)
        if score > best_score:
            best_score = score
            best_result = circles.copy()
            best_W, best_H = W, H

        # Try random restarts if time permits
        remaining = DEADLINE - 10 - (time.time() - start_time)
        if remaining > 5:
            n_restarts = max(1, int(remaining / 6))
            for _ in range(n_restarts):
                if time.time() - start_time > DEADLINE - 10:
                    break
                circ_rand = _random_initialization(W, H, rng)
                circ_opt = _local_optimize(circ_rand, W, H, max_iter=1000)
                circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
                circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
                circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)
                circ_opt, valid_rand = _separate_circles(circ_opt, W, H)
                if valid_rand:
                    s = np.sum(circ_opt[:, 2])
                    if s > best_score:
                        best_score = s
                        best_result = circ_opt.copy()
                        best_W, best_H = W, H

    # Final high-precision polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 12:
        polished = _polish(best_result, best_W, best_H, max_iter=12000)
        polished[:, 0] = np.clip(polished[:, 0], polished[:, 2], best_W - polished[:, 2])
        polished[:, 1] = np.clip(polished[:, 1], polished[:, 2], best_H - polished[:, 2])
        polished[:, 2] = np.clip(polished[:, 2], 0.001, min(best_W, best_H) / 2)
        polished, valid_final = _separate_circles(polished, best_W, best_H, max_iter=1500, tol=1e-14)
        if valid_final:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Second polish pass with ultra-tight settings if time permits
    if best_result is not None and time.time() - start_time < DEADLINE - 5:
        polished2 = _polish(best_result, best_W, best_H, max_iter=6000)
        polished2[:, 0] = np.clip(polished2[:, 0], polished2[:, 2], best_W - polished2[:, 2])
        polished2[:, 1] = np.clip(polished2[:, 1], polished2[:, 2], best_H - polished2[:, 2])
        polished2[:, 2] = np.clip(polished2[:, 2], 0.001, min(best_W, best_H) / 2)
        polished2, valid_final2 = _separate_circles(polished2, best_W, best_H, max_iter=2000, tol=1e-14)
        if valid_final2:
            s = np.sum(polished2[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished2.copy()

    # Final fallback: simple grid in square
    if best_result is None:
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    # Ensure the result is valid; if not, fall back to a safe grid
    if not _check_validity(best_result, best_W, best_H):
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                r = 0.5 / side - 1e-3
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])
        best_W, best_H = 1.0, 1.0

    return best_result
# EVOLVE-BLOCK-END
