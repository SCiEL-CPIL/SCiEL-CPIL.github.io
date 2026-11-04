"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii.

Structural change: reformulate as a discrete search over aspect ratio using
Nelder-Mead on a derived objective.  For a given aspect ratio we build a
hexagonal lattice of candidate centres, compute the maximal feasible radius
of each centre (distance to walls and to every other centre), then greedily
pick the N centres with the largest radii.  This yields a valid packing whose
quality is a (piecewise smooth) function of the aspect ratio, which we
optimise with a derivative-free global method.  We keep the continuous
L-BFGS-B polish as a final refinement step.
"""
import numpy as np
import time
from scipy.optimize import minimize, minimize as _scipy_minimize

N = 21
DEADLINE = 140.0


def _check_validity(circles, W, H, tol=1e-9):
    """Check if circles are valid: inside rectangle, no overlaps."""
    if len(circles) != N:
        return False
    for i in range(N):
        x, y, r = circles[i]
        if r <= 0:
            return False
        if x - r < -tol or y - r < -tol or x + r > W + tol or y + r > H + tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            dx = circles[i][0] - circles[j][0]
            dy = circles[i][1] - circles[j][1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < circles[i][2] + circles[j][2] - tol:
                return False
    return True


def _separate_circles(circles, W, H, max_iter=3000, tol=1e-12):
    """Directly resolve overlaps and boundary violations using pairwise repulsion."""
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    for it in range(max_iter):
        moved = False
        for i in range(n):
            if r[i] <= 0:
                r[i] = 1e-4
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])
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
                    push = overlap + 1e-8
                    x[i] += wi * push * dx / dist
                    y[i] += wi * push * dy / dist
                    x[j] -= wj * push * dx / dist
                    y[j] -= wj * push * dy / dist
                    moved = True
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


def _max_radius_at(x, y, W, H, other_x, other_y, other_r, exclude_idx=None):
    """Maximum radius a circle of centre (x,y) can have, given other circles."""
    max_r = min(x, W - x, y, H - y)
    if other_x is not None and len(other_x) > 0:
        for k in range(len(other_x)):
            if exclude_idx is not None and k == exclude_idx:
                continue
            d = np.sqrt((x - other_x[k]) ** 2 + (y - other_y[k]) ** 2)
            max_r = min(max_r, d - other_r[k])
    return max(1e-8, max_r)


def _hex_candidate_grid(W, H, r_base):
    """Generate a dense hexagonal lattice of candidate centres."""
    dx = 2 * r_base * 1.0001
    dy = 2 * r_base * np.sqrt(3) / 2 * 1.0001
    candidates = []
    row = 0
    y_pos = r_base
    while y_pos <= H - r_base + 1e-9:
        offset = 0.0 if row % 2 == 0 else dx / 2
        x_pos = r_base + offset
        while x_pos <= W - r_base + 1e-9:
            candidates.append((x_pos, y_pos))
            x_pos += dx
        y_pos += dy
        row += 1
    # Also add a few boundary-aligned candidates to capture edge effects
    for i in range(max(1, int(W / (2 * r_base)))):
        x = (i + 0.5) * W / max(1, int(W / (2 * r_base)))
        candidates.append((x, r_base))
        candidates.append((x, H - r_base))
    for j in range(max(1, int(H / (2 * r_base)))):
        y = (j + 0.5) * H / max(1, int(H / (2 * r_base)))
        candidates.append((r_base, y))
        candidates.append((W - r_base, y))
    # Deduplicate
    unique = []
    seen = set()
    for cx, cy in candidates:
        key = (round(cx, 6), round(cy, 6))
        if key not in seen:
            seen.add(key)
            unique.append((cx, cy))
    return np.array(unique)


def _greedy_pack_from_candidates(W, H, r_base):
    """
    Build a valid packing of N circles by greedily selecting the best
    candidate centres from a hexagonal grid.  The radius of each selected
    circle is the minimum distance to walls and to already-selected circles.
    """
    candidates = _hex_candidate_grid(W, H, r_base)
    if len(candidates) == 0:
        # Fallback: uniform grid
        side = 5
        candidates = np.array([( (i+0.5)*W/side, (j+0.5)*H/side )
                               for i in range(side) for j in range(side)])

    selected = []
    remaining = candidates.copy()
    for _ in range(N):
        if len(remaining) == 0:
            break
        best_idx = 0
        best_r = -1
        other_x = np.array([c[0] for c in selected]) if selected else None
        other_y = np.array([c[1] for c in selected]) if selected else None
        other_r = np.array([c[2] for c in selected]) if selected else None
        for k in range(len(remaining)):
            cx, cy = remaining[k]
            r = _max_radius_at(cx, cy, W, H, other_x, other_y, other_r)
            if r > best_r:
                best_r = r
                best_idx = k
        if best_r <= 1e-8:
            break
        cx, cy = remaining[best_idx]
        selected.append((cx, cy, best_r))
        remaining = np.delete(remaining, best_idx, axis=0)

    if len(selected) < N:
        # Fill remaining with tiny circles in free space
        for _ in range(N - len(selected)):
            best_x, best_y, best_r = W / 2, H / 2, 1e-4
            for _ in range(200):
                tx = np.random.uniform(0.01, W - 0.01)
                ty = np.random.uniform(0.01, H - 0.01)
                r = _max_radius_at(tx, ty, W, H,
                                   np.array([c[0] for c in selected]) if selected else None,
                                   np.array([c[1] for c in selected]) if selected else None,
                                   np.array([c[2] for c in selected]) if selected else None)
                if r > best_r:
                    best_r = r
                    best_x, best_y = tx, ty
            selected.append((best_x, best_y, max(best_r, 1e-4)))

    return np.array(selected[:N])


def _greedy_pack_score(W, H, r_base=0.12):
    """Return the sum of radii of the greedy packing for given W, H."""
    circles = _greedy_pack_from_candidates(W, H, r_base)
    if not _check_validity(circles, W, H):
        circles, valid = _separate_circles(circles, W, H, max_iter=500)
        if not valid:
            return -np.inf
    return np.sum(circles[:, 2])


def _objective_aspect(ar, r_base=0.12):
    """Objective for Nelder-Mead: minimize negative sum of radii."""
    W = 2.0 * ar / (ar + 1.0)
    H = 2.0 / (ar + 1.0)
    s = _greedy_pack_score(W, H, r_base)
    if np.isinf(s):
        return 1e6
    return -s


def _local_optimize(circles, W, H, max_iter=3000):
    """L-BFGS-B with smooth-max penalties."""
    n = N
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    params = np.concatenate([x, y, r])
    r_max = min(W, H) / 2
    bounds = []
    for i in range(n):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    def _obj_and_grad(params, W, H, eps, bw, ow):
        n = N
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2*n:3*n] = -1.0

        def _soft_max(v, scale):
            v_safe = np.clip(v / scale, -50, 50)
            smax = scale * np.log1p(np.exp(v_safe))
            sig = 1.0 / (1.0 + np.exp(-v_safe))
            return smax, sig

        for (vi, gi) in [(r - x, 0), (r - y, n), (x + r - W, 0), (y + r - H, n), (-r, 2*n)]:
            smax, gsmax = _soft_max(vi, eps)
            obj += bw * np.sum(smax ** 2)
            if gi == 0:
                grad[0:n] += -2.0 * bw * smax * gsmax
                grad[2*n:3*n] += 2.0 * bw * smax * gsmax
            elif gi == n:
                grad[n:2*n] += -2.0 * bw * smax * gsmax
                grad[2*n:3*n] += 2.0 * bw * smax * gsmax
            else:
                grad[2*n:3*n] += -2.0 * bw * smax * gsmax

        dx_mat = x[:, None] - x[None, :]
        dy_mat = y[:, None] - y[None, :]
        dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
        np.fill_diagonal(dist_mat, 1e12)
        dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)
        iu, ju = np.triu_indices(n, k=1)
        dist_ij = dist_mat_safe[iu, ju]
        dx_ij = dx_mat[iu, ju]
        dy_ij = dy_mat[iu, ju]
        r_sum = r[iu] + r[ju]
        v_vec = r_sum - dist_ij
        smax_vec, gsmax_vec = _soft_max(v_vec, eps)
        obj += ow * np.sum(smax_vec ** 2)
        factor_vec = 2.0 * ow * smax_vec * gsmax_vec
        nx_ij = -dx_ij / dist_ij
        ny_ij = -dy_ij / dist_ij
        np.add.at(grad, iu, factor_vec * nx_ij)
        np.add.at(grad, n + iu, factor_vec * ny_ij)
        np.add.at(grad, 2*n + iu, factor_vec)
        np.add.at(grad, ju, -factor_vec * nx_ij)
        np.add.at(grad, n + ju, -factor_vec * ny_ij)
        np.add.at(grad, 2*n + ju, factor_vec)
        return obj, grad

    result = _scipy_minimize(_obj_and_grad, params, args=(W, H, 5e-4, 150.0, 150.0),
                             method='L-BFGS-B', jac=True, bounds=bounds,
                             options={'maxiter': max_iter // 2, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 200})
    params = result.x
    result = _scipy_minimize(_obj_and_grad, params, args=(W, H, 1e-5, 400.0, 400.0),
                             method='L-BFGS-B', jac=True, bounds=bounds,
                             options={'maxiter': max_iter // 2, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 300})
    return result.x.reshape(N, 3)


def _polish(circles, W, H, max_iter=8000):
    """Multi-pass polish."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    def _obj_and_grad(params, W, H, eps, bw, ow):
        n = N
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2*n:3*n] = -1.0

        def _soft_max(v, scale):
            v_safe = np.clip(v / scale, -50, 50)
            smax = scale * np.log1p(np.exp(v_safe))
            sig = 1.0 / (1.0 + np.exp(-v_safe))
            return smax, sig

        for (vi, gi) in [(r - x, 0), (r - y, n), (x + r - W, 0), (y + r - H, n), (-r, 2*n)]:
            smax, gsmax = _soft_max(vi, eps)
            obj += bw * np.sum(smax ** 2)
            if gi == 0:
                grad[0:n] += -2.0 * bw * smax * gsmax
                grad[2*n:3*n] += 2.0 * bw * smax * gsmax
            elif gi == n:
                grad[n:2*n] += -2.0 * bw * smax * gsmax
                grad[2*n:3*n] += 2.0 * bw * smax * gsmax
            else:
                grad[2*n:3*n] += -2.0 * bw * smax * gsmax

        dx_mat = x[:, None] - x[None, :]
        dy_mat = y[:, None] - y[None, :]
        dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
        np.fill_diagonal(dist_mat, 1e12)
        dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)
        iu, ju = np.triu_indices(n, k=1)
        dist_ij = dist_mat_safe[iu, ju]
        dx_ij = dx_mat[iu, ju]
        dy_ij = dy_mat[iu, ju]
        r_sum = r[iu] + r[ju]
        v_vec = r_sum - dist_ij
        smax_vec, gsmax_vec = _soft_max(v_vec, eps)
        obj += ow * np.sum(smax_vec ** 2)
        factor_vec = 2.0 * ow * smax_vec * gsmax_vec
        nx_ij = -dx_ij / dist_ij
        ny_ij = -dy_ij / dist_ij
        np.add.at(grad, iu, factor_vec * nx_ij)
        np.add.at(grad, n + iu, factor_vec * ny_ij)
        np.add.at(grad, 2*n + iu, factor_vec)
        np.add.at(grad, ju, -factor_vec * nx_ij)
        np.add.at(grad, n + ju, -factor_vec * ny_ij)
        np.add.at(grad, 2*n + ju, factor_vec)
        return obj, grad

    for eps_val, w_val in [(1e-3, 200.0), (5e-4, 300.0), (1e-4, 400.0), (5e-5, 500.0), (1e-5, 600.0), (5e-6, 800.0)]:
        result = _scipy_minimize(_obj_and_grad, params, args=(W, H, eps_val, w_val, w_val),
                                 method='L-BFGS-B', jac=True, bounds=bounds,
                                 options={'maxiter': max_iter // 6, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 300})
        params = result.x
    return result.x.reshape(N, 3)


def _high_precision_polish(circles, W, H, max_iter=12000):
    """Ultra-tight final polish."""
    params = circles.flatten()
    r_max = min(W, H) / 2
    bounds = []
    for i in range(N):
        bounds.append((1e-6, W - 1e-6))
        bounds.append((1e-6, H - 1e-6))
        bounds.append((1e-6, r_max))

    def _obj_and_grad(params, W, H, eps, bw, ow):
        n = N
        x = params[0:n]
        y = params[n:2*n]
        r = params[2*n:3*n]
        obj = -np.sum(r)
        grad = np.zeros_like(params)
        grad[2*n:3*n] = -1.0

        def _soft_max(v, scale):
            v_safe = np.clip(v / scale, -50, 50)
            smax = scale * np.log1p(np.exp(v_safe))
            sig = 1.0 / (1.0 + np.exp(-v_safe))
            return smax, sig

        for (vi, gi) in [(r - x, 0), (r - y, n), (x + r - W, 0), (y + r - H, n), (-r, 2*n)]:
            smax, gsmax = _soft_max(vi, eps)
            obj += bw * np.sum(smax ** 2)
            if gi == 0:
                grad[0:n] += -2.0 * bw * smax * gsmax
                grad[2*n:3*n] += 2.0 * bw * smax * gsmax
            elif gi == n:
                grad[n:2*n] += -2.0 * bw * smax * gsmax
                grad[2*n:3*n] += 2.0 * bw * smax * gsmax
            else:
                grad[2*n:3*n] += -2.0 * bw * smax * gsmax

        dx_mat = x[:, None] - x[None, :]
        dy_mat = y[:, None] - y[None, :]
        dist_mat = np.sqrt(dx_mat ** 2 + dy_mat ** 2)
        np.fill_diagonal(dist_mat, 1e12)
        dist_mat_safe = np.where(dist_mat < 1e-12, 1e-12, dist_mat)
        iu, ju = np.triu_indices(n, k=1)
        dist_ij = dist_mat_safe[iu, ju]
        dx_ij = dx_mat[iu, ju]
        dy_ij = dy_mat[iu, ju]
        r_sum = r[iu] + r[ju]
        v_vec = r_sum - dist_ij
        smax_vec, gsmax_vec = _soft_max(v_vec, eps)
        obj += ow * np.sum(smax_vec ** 2)
        factor_vec = 2.0 * ow * smax_vec * gsmax_vec
        nx_ij = -dx_ij / dist_ij
        ny_ij = -dy_ij / dist_ij
        np.add.at(grad, iu, factor_vec * nx_ij)
        np.add.at(grad, n + iu, factor_vec * ny_ij)
        np.add.at(grad, 2*n + iu, factor_vec)
        np.add.at(grad, ju, -factor_vec * nx_ij)
        np.add.at(grad, n + ju, -factor_vec * ny_ij)
        np.add.at(grad, 2*n + ju, factor_vec)
        return obj, grad

    for eps_val, w_val in [(1e-5, 600.0), (1e-6, 800.0), (5e-7, 1000.0)]:
        result = _scipy_minimize(_obj_and_grad, params, args=(W, H, eps_val, w_val, w_val),
                                 method='L-BFGS-B', jac=True, bounds=bounds,
                                 options={'maxiter': max_iter // 3, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 400})
        params = result.x
    return result.x.reshape(N, 3)


def _safe_clip(circles, W, H):
    circles = circles.copy()
    circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], W - circles[:, 2])
    circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], H - circles[:, 2])
    circles[:, 2] = np.clip(circles[:, 2], 0.001, min(W, H) / 2)
    return circles


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    best_W, best_H = 1.0, 1.0

    # Phase 1: Nelder-Mead over aspect ratio using greedy hexagonal packing
    # This is the structural change: discrete candidate positions + derivative-free global search
    aspect_starts = [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.20, 1.35]
    r_base = 0.10  # base radius for hex grid spacing

    for ar0 in aspect_starts:
        if time.time() - start_time > DEADLINE - 40:
            break
        try:
            result = _scipy_minimize(
                _objective_aspect, [ar0], args=(r_base,),
                method='Nelder-Mead',
                options={'xatol': 1e-5, 'fatol': 1e-8, 'maxiter': 200, 'adaptive': True}
            )
            ar_opt = result.x[0]
            W = 2.0 * ar_opt / (ar_opt + 1.0)
            H = 2.0 / (ar_opt + 1.0)
            circles = _greedy_pack_from_candidates(W, H, r_base)
            circles, valid = _separate_circles(circles, W, H, max_iter=1000)
            if valid:
                s = np.sum(circles[:, 2])
                if s > best_score:
                    best_score = s
                    best_result = circles.copy()
                    best_W, best_H = W, H
        except Exception:
            pass

    # Phase 2: Also try a direct grid of aspect ratios with greedy packing
    for ar in [0.80, 0.85, 0.90, 0.94, 0.97, 1.0, 1.03, 1.06, 1.10, 1.15, 1.25, 1.40, 1.60]:
        if time.time() - start_time > DEADLINE - 30:
            break
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)
        circles = _greedy_pack_from_candidates(W, H, r_base)
        circles, valid = _separate_circles(circles, W, H, max_iter=1000)
        if valid:
            s = np.sum(circles[:, 2])
            if s > best_score:
                best_score = s
                best_result = circles.copy()
                best_W, best_H = W, H

    # Phase 3: Continuous L-BFGS-B polish on the best result
    if best_result is not None and time.time() - start_time < DEADLINE - 25:
        polished = _local_optimize(best_result, best_W, best_H, max_iter=3000)
        polished = _safe_clip(polished, best_W, best_H)
        polished, valid = _separate_circles(polished, best_W, best_H, max_iter=1000)
        if valid:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Phase 4: Multi-pass polish
    if best_result is not None and time.time() - start_time < DEADLINE - 15:
        polished = _polish(best_result, best_W, best_H, max_iter=8000)
        polished = _safe_clip(polished, best_W, best_H)
        polished, valid = _separate_circles(polished, best_W, best_H, max_iter=1000)
        if valid:
            s = np.sum(polished[:, 2])
            if s > best_score:
                best_score = s
                best_result = polished.copy()

    # Phase 5: High-precision final polish
    if best_result is not None and time.time() - start_time < DEADLINE - 8:
        hp = _high_precision_polish(best_result, best_W, best_H, max_iter=10000)
        hp = _safe_clip(hp, best_W, best_H)
        hp, valid = _separate_circles(hp, best_W, best_H, max_iter=2000)
        if valid:
            s = np.sum(hp[:, 2])
            if s > best_score:
                best_score = s
                best_result = hp.copy()

    # Fallback
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
