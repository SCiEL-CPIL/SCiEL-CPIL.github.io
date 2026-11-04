"""Circle packing in perimeter-4 rectangle: pack 21 circles, maximize sum of radii.
Structural change: physics-based iterative relaxation with dynamic contact graph,
combined with SLSQP refinement and multiple aspect-ratio search.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 21

# EVOLVE-BLOCK-START

def _validate_solution(pos_r, w, h, n_circles, tol=1e-7):
    """Check if circles are valid: disjoint interiors, within bounds."""
    if pos_r.shape[0] != n_circles:
        return False
    xs = pos_r[:, 0]
    ys = pos_r[:, 1]
    rs = pos_r[:, 2]
    if np.any(rs < 1e-9):
        return False
    if np.any(xs - rs < -tol) or np.any(xs + rs > w + tol):
        return False
    if np.any(ys - rs < -tol) or np.any(ys + rs > h + tol):
        return False
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < (rs[i] + rs[j]) - tol:
                return False
    return True


def _resolve_collisions(circles, w, h, max_iters=3000, tol=1e-9):
    """Relaxation to resolve overlaps and boundary violations."""
    n = circles.shape[0]
    pos = circles[:, :2].copy()
    rad = circles[:, 2].copy()

    for _ in range(max_iters):
        moved = False
        # Boundary constraints
        for i in range(n):
            x, y, r = pos[i, 0], pos[i, 1], rad[i]
            if x - r < 0:
                pos[i, 0] = r
                moved = True
            if x + r > w:
                pos[i, 0] = w - r
                moved = True
            if y - r < 0:
                pos[i, 1] = r
                moved = True
            if y + r > h:
                pos[i, 1] = h - r
                moved = True
        # Pairwise collisions
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = rad[i] + rad[j]
                if dist < min_dist:
                    if dist < 1e-9:
                        angle = np.random.uniform(0, 2 * np.pi)
                        push = (min_dist - dist) * 0.5
                        pos[i, 0] -= np.cos(angle) * push
                        pos[i, 1] -= np.sin(angle) * push
                        pos[j, 0] += np.cos(angle) * push
                        pos[j, 1] += np.sin(angle) * push
                        moved = True
                    else:
                        overlap = min_dist - dist
                        nx = dx / dist
                        ny = dy / dist
                        pos[i, 0] -= nx * overlap * 0.5
                        pos[i, 1] -= ny * overlap * 0.5
                        pos[j, 0] += nx * overlap * 0.5
                        pos[j, 1] += ny * overlap * 0.5
                        moved = True
        if not moved:
            break

    # Final clamp
    for i in range(n):
        pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
        pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
    return np.column_stack((pos, rad))


def _generate_grid_init(w, h, n_circles, cols, rows):
    """Generate initial circle positions in a grid pattern."""
    circles = np.zeros((n_circles, 3))
    count = 0
    for i in range(cols):
        for j in range(rows):
            if count >= n_circles:
                break
            x = (i + 0.5) * w / cols
            y = (j + 0.5) * h / rows
            cell_w = w / cols
            cell_h = h / rows
            r = 0.45 * min(cell_w, cell_h)
            circles[count, 0] = x
            circles[count, 1] = y
            circles[count, 2] = r
            count += 1
    return circles


def _slsqp_optimize(circles, w, h, deadline, maxiter=200):
    """Optimize radii and positions using SLSQP with hard constraints."""
    if time.time() > deadline - 5.0:
        return None

    n = len(circles)
    x0 = circles.flatten()

    def objective(x):
        if time.time() > deadline:
            return 0.0
        r = x[2::3]
        return -np.sum(r)

    constraints = []
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i * 3] - x[i * 3 + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: w - x[i * 3] - x[i * 3 + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i * 3 + 1] - x[i * 3 + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: h - x[i * 3 + 1] - x[i * 3 + 2]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i * 3 + 2] - 1e-6})

    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: np.sqrt((x[i * 3] - x[j * 3]) ** 2 +
                                                    (x[i * 3 + 1] - x[j * 3 + 1]) ** 2)
                                        - x[i * 3 + 2] - x[j * 3 + 2]
            })

    bounds = []
    for i in range(n):
        bounds.append((1e-6, w))
        bounds.append((1e-6, h))
        bounds.append((1e-6, min(w, h) / 2))

    try:
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': maxiter, 'ftol': 1e-9}
        )
        if res.success:
            new_circles = res.x.reshape(-1, 3)
            if _validate_solution(new_circles, w, h, N):
                return new_circles
            else:
                fixed = _resolve_collisions(new_circles, w, h, max_iters=100)
                if _validate_solution(fixed, w, h, N):
                    return fixed
    except Exception:
        pass
    return None


def _expand_uniform(circles, w, h, deadline, steps=50):
    """Expand uniform radii iteratively with relaxation between steps."""
    if time.time() > deadline - 5.0:
        return circles
    r = circles[0, 2]
    for step in range(steps):
        if time.time() > deadline - 5.0:
            break
        r_new = r * 1.01
        circles[:, 2] = r_new
        fixed = _resolve_collisions(circles, w, h, max_iters=2000)
        if _validate_solution(fixed, w, h, N):
            circles = fixed
            r = r_new
        else:
            break
    return circles


def _solve_aspect(w, h, deadline):
    """Solve for a specific aspect ratio using grid init + relaxation + SLSQP."""
    ar = w / h
    if ar > 1.5:
        grid_configs = [(4, 5), (3, 7), (4, 6), (5, 4)]
    elif ar < 0.67:
        grid_configs = [(5, 4), (7, 3), (6, 4), (4, 5)]
    else:
        grid_configs = [(4, 5), (5, 4), (5, 5), (4, 6), (6, 4), (3, 7), (7, 3)]

    best_circles = None
    best_score = 0.0

    for cols, rows in grid_configs:
        if cols * rows < N:
            continue
        init_circles = _generate_grid_init(w, h, N, cols, rows)
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=2000)
        if not _validate_solution(relaxed, w, h, N):
            continue

        # Expand radii
        expanded = _expand_uniform(relaxed, w, h, deadline)
        score = np.sum(expanded[:, 2])
        if score > best_score:
            best_score = score
            best_circles = expanded.copy()

        # SLSQP refinement
        if time.time() < deadline - 10.0:
            refined = _slsqp_optimize(expanded, w, h, deadline, maxiter=200)
            if refined is not None:
                refined_score = np.sum(refined[:, 2])
                if refined_score > best_score:
                    best_score = refined_score
                    best_circles = refined.copy()

    if best_circles is None:
        # Fallback
        init_circles = _generate_grid_init(w, h, N, 5, 5)
        relaxed = _resolve_collisions(init_circles, w, h, max_iters=2000)
        if _validate_solution(relaxed, w, h, N):
            best_circles = relaxed
            best_score = np.sum(relaxed[:, 2])

    return best_circles, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20 second margin for safety

    best_result = None
    best_score = -np.inf

    # Try different aspect ratios
    aspect_ratios = [1.0, 1.2, 0.83, 1.5, 0.67, 2.0, 0.5, 1.33, 0.75, 1.1, 0.9, 1.4, 0.7]

    for ar in aspect_ratios:
        if time.time() > deadline:
            break

        w = 2.0 * ar / (1.0 + ar)
        h = 2.0 / (1.0 + ar)

        result, score = _solve_aspect(w, h, deadline)
        if result is not None and score > best_score:
            best_score = score
            best_result = result

    if best_result is not None:
        return best_result
    else:
        # Fallback: simple grid in unit square (perimeter = 4)
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

# EVOLVE-BLOCK-END
