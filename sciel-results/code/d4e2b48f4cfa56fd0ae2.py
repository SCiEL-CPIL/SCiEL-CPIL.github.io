"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _eval_packing(xy, r):
    n = len(xy)
    for i in range(n):
        x, y, ri = xy[i, 0], xy[i, 1], r[i]
        if ri < -1e-9:
            return 0.0, False
        if x - ri < -1e-9 or 1.0 - x - ri < -1e-9 or y - ri < -1e-9 or 1.0 - y - ri < -1e-9:
            return 0.0, False
    for i in range(n):
        for j in range(i + 1, n):
            dx = xy[i, 0] - xy[j, 0]
            dy = xy[i, 1] - xy[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return 0.0, False
    return float(np.sum(r)), True


def _largest_empty_circle(circles_xy, circles_r, n_samples=300):
    """
    Find the largest empty circle among existing circles and walls.
    Uses circle-circle intersection points + random samples as candidate centers.
    Returns (x, y, radius).
    """
    n = len(circles_xy)
    best_r = 0.0
    best_x, best_y = 0.5, 0.5

    # Candidate centers: circle-circle intersections, circle centers, random points
    candidates = []

    # Circle-circle intersection points
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles_xy[j, 0] - circles_xy[i, 0]
            dy = circles_xy[j, 1] - circles_xy[i, 1]
            d = np.sqrt(dx * dx + dy * dy)
            ri, rj = circles_r[i], circles_r[j]
            if d < 1e-12:
                continue
            # Only if circles are close enough for intersections to be relevant
            if d > ri + rj + 0.1:
                continue
            # Intersection of two circles
            a = (ri * ri - rj * rj + d * d) / (2 * d)
            h_sq = ri * ri - a * a
            if h_sq < 0:
                continue
            h = np.sqrt(h_sq)
            mx = circles_xy[i, 0] + a * dx / d
            my = circles_xy[i, 1] + a * dy / d
            # Perpendicular direction
            px = -dy / d
            py = dx / d
            candidates.append([mx + h * px, my + h * py])
            candidates.append([mx - h * px, my - h * py])

    # Also add circle centers
    for i in range(n):
        candidates.append([circles_xy[i, 0], circles_xy[i, 1]])

    # Random samples
    rng = np.random.RandomState(42)
    for _ in range(n_samples):
        candidates.append([rng.uniform(0.01, 0.99), rng.uniform(0.01, 0.99)])

    candidates = np.array(candidates)

    for cx, cy in candidates:
        # Wall clearance
        wall_r = min(cx, 1.0 - cx, cy, 1.0 - cy)
        if wall_r < 1e-6:
            continue
        # Circle clearance
        if n > 0:
            dx = circles_xy[:, 0] - cx
            dy = circles_xy[:, 1] - cy
            dists = np.sqrt(dx * dx + dy * dy)
            circle_r = np.min(dists - circles_r)
        else:
            circle_r = 0.5
        r = min(wall_r, circle_r)
        if r > best_r:
            best_r = r
            best_x, best_y = cx, cy

    return best_x, best_y, best_r


def _greedy_packing(deadline, n_circles=N, seed=0):
    """
    Greedy packing: iteratively place the largest possible empty circle.
    Returns (xy, r, total).
    """
    rng = np.random.RandomState(seed)
    circles_xy = []
    circles_r = []

    # Start with a few random small circles to break symmetry
    for _ in range(3):
        x = rng.uniform(0.2, 0.8)
        y = rng.uniform(0.2, 0.8)
        r = 0.02
        circles_xy.append([x, y])
        circles_r.append(r)

    circles_xy = np.array(circles_xy)
    circles_r = np.array(circles_r)

    for k in range(n_circles - len(circles_r)):
        if time.time() > deadline:
            break
        # Find largest empty circle
        ex, ey, er = _largest_empty_circle(circles_xy, circles_r, n_samples=200)
        if er < 1e-4:
            # Can't fit more; place tiny circle
            ex, ey = rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95)
            er = 0.005
        circles_xy = np.vstack([circles_xy, [[ex, ey]]])
        circles_r = np.append(circles_r, er)

    total = float(np.sum(circles_r))
    return circles_xy, circles_r, total


def _repulsion_relax(xy, r, deadline, max_iter=500):
    """
    Pairwise repulsion: iteratively push overlapping circles apart.
    Also adjust radii to be as large as possible given neighbors.
    """
    n = len(xy)
    xy = xy.copy()
    r = r.copy()
    best_total = 0.0
    best_valid = False

    for iteration in range(max_iter):
        if time.time() > deadline:
            break

        # Check validity and record best
        total, valid = _eval_packing(xy, r)
        if valid and total > best_total:
            best_total = total
            best_valid = True

        if not valid:
            # Resolve overlaps by pushing apart
            for i in range(n):
                for j in range(i + 1, n):
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    overlap = r[i] + r[j] - dist
                    if overlap > 0:
                        if dist < 1e-12:
                            # Random push
                            angle = np.random.uniform(0, 2 * np.pi)
                            dx, dy = np.cos(angle), np.sin(angle)
                            dist = 1e-12
                        push = overlap * 0.5
                        xy[i, 0] -= push * dx / dist
                        xy[i, 1] -= push * dy / dist
                        xy[j, 0] += push * dx / dist
                        xy[j, 1] += push * dy / dist

            # Clamp to walls
            for i in range(n):
                xy[i, 0] = np.clip(xy[i, 0], 0.0, 1.0)
                xy[i, 1] = np.clip(xy[i, 1], 0.0, 1.0)
                # Reduce radius if hitting wall
                wall_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                if r[i] > wall_r:
                    r[i] = wall_r
        else:
            # Valid: try to grow each circle
            for i in range(n):
                # Max radius for circle i given others
                wall_r = min(xy[i, 0], 1.0 - xy[i, 0], xy[i, 1], 1.0 - xy[i, 1])
                max_r = wall_r
                for j in range(n):
                    if j == i:
                        continue
                    dx = xy[j, 0] - xy[i, 0]
                    dy = xy[j, 1] - xy[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    max_r = min(max_r, dist - r[j])
                if max_r > r[i] + 1e-10:
                    r[i] = max_r

    # Final validation
    total, valid = _eval_packing(xy, r)
    if valid and total > best_total:
        best_total = total
        best_valid = True

    return xy, r, best_total, best_valid


def _gradient_polish(xy, r, deadline, tol=1e-8, max_iter=300):
    """
    Gradient-based polish using L-BFGS-B on a smooth objective.
    """
    n = len(xy)
    v0 = np.concatenate([xy[:, 0], xy[:, 1], r])

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.999))
    for i in range(n):
        bounds.append((0.001, 0.49))

    def obj(v):
        if time.time() > deadline:
            return float('inf')
        x = v[0:n]
        y = v[n:2*n]
        rr = v[2*n:3*n]
        # Wall violations
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = rr - wall_min
        beta = 20.0
        wall_pen = np.sum(np.log1p(np.exp(beta * wall_viol)) / beta)
        # Pairwise
        total_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                viol = rr[i] + rr[j] - dist
                total_viol += np.log1p(np.exp(beta * viol)) / beta
        return -np.sum(rr) + 100.0 * (wall_pen + total_viol)

    def grad(v):
        if time.time() > deadline:
            return np.zeros(3 * n)
        x = v[0:n].copy()
        y = v[n:2*n].copy()
        rr = v[2*n:3*n].copy()
        g = np.zeros(3 * n)
        beta = 20.0
        # Wall
        wall_min = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        wall_viol = rr - wall_min
        sigmoid_w = 1.0 / (1.0 + np.exp(-beta * wall_viol))
        for i in range(n):
            sw = sigmoid_w[i]
            if sw > 1e-10:
                dists = [x[i], 1.0 - x[i], y[i], 1.0 - y[i]]
                min_idx = int(np.argmin(dists))
                g[2*n + i] += 100.0 * sw
                if min_idx == 0:
                    g[i] -= 100.0 * sw
                elif min_idx == 1:
                    g[i] += 100.0 * sw
                elif min_idx == 2:
                    g[n + i] -= 100.0 * sw
                else:
                    g[n + i] += 100.0 * sw
        # Pairwise
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                viol = rr[i] + rr[j] - dist
                s = 1.0 / (1.0 + np.exp(-beta * viol))
                if s > 1e-10:
                    factor = 100.0 * s
                    if dist > 1e-12:
                        g[i] += factor * dx / dist
                        g[n + i] += factor * dy / dist
                        g[j] -= factor * dx / dist
                        g[n + j] -= factor * dy / dist
                    g[2*n + i] += factor
                    g[2*n + j] += factor
        g[2*n:3*n] -= 1.0
        return g

    try:
        result = minimize(obj, v0, jac=grad, method='L-BFGS-B',
                          bounds=bounds,
                          options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-6,
                                   'maxfun': 8000})
    except Exception:
        return xy, r

    v_opt = result.x
    new_xy = np.column_stack([v_opt[0:n], v_opt[n:2*n]])
    new_r = v_opt[2*n:3*n]
    total, valid = _eval_packing(new_xy, new_r)
    if valid:
        return new_xy, new_r
    return xy, r


def _simulated_annealing(xy, r, deadline, n_iter=2000):
    """
    Simulated annealing: perturb positions and radii, accept with probability based on energy.
    Energy = -sum(r) + penalty for violations.
    """
    n = len(xy)
    best_xy = xy.copy()
    best_r = r.copy()
    best_total, best_valid = _eval_packing(xy, r)
    if best_valid:
        best_total = float(np.sum(best_r))
    else:
        best_total = 0.0

    def energy(xy_curr, r_curr):
        total_r = np.sum(r_curr)
        # Wall penalty
        wall_pen = 0.0
        for i in range(n):
            wall_min = min(xy_curr[i, 0], 1.0 - xy_curr[i, 0], xy_curr[i, 1], 1.0 - xy_curr[i, 1])
            if r_curr[i] > wall_min:
                wall_pen += (r_curr[i] - wall_min) ** 2
        # Pairwise penalty
        pair_pen = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = xy_curr[i, 0] - xy_curr[j, 0]
                dy = xy_curr[i, 1] - xy_curr[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r_curr[i] + r_curr[j]:
                    pair_pen += (r_curr[i] + r_curr[j] - dist) ** 2
        return -total_r + 10.0 * (wall_pen + pair_pen)

    T_start = 0.05
    T_end = 0.0001
    current_e = energy(xy, r)

    for it in range(n_iter):
        if time.time() > deadline:
            break
        T = T_start * (T_end / T_start) ** (it / n_iter)

        # Perturb: move one or two circles
        n_perturb = np.random.randint(1, 3)
        idxs = np.random.choice(n, n_perturb, replace=False)
        new_xy = xy.copy()
        new_r = r.copy()
        for i in idxs:
            new_xy[i, 0] += np.random.normal(0, 0.02)
            new_xy[i, 1] += np.random.normal(0, 0.02)
            new_r[i] += np.random.normal(0, 0.01)
            new_xy[i, 0] = np.clip(new_xy[i, 0], 0.001, 0.999)
            new_xy[i, 1] = np.clip(new_xy[i, 1], 0.001, 0.999)
            new_r[i] = np.clip(new_r[i], 0.001, 0.49)

        new_e = energy(new_xy, new_r)
        delta_e = new_e - current_e

        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            xy, r = new_xy, new_r
            current_e = new_e
            total, valid = _eval_packing(xy, r)
            if valid and total > best_total:
                best_total = total
                best_valid = True
                best_xy = xy.copy()
                best_r = r.copy()

    return best_xy, best_r, best_total, best_valid


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    best_xy = None
    best_r = None
    best_total = 0.0
    best_valid = False

    # Phase 1: Multiple greedy packings with different seeds
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        xy, r, total = _greedy_packing(deadline - 30, n_circles=N, seed=seed)
        # Relax
        rel_xy, rel_r, rel_total, rel_valid = _repulsion_relax(xy, r, deadline - 25, max_iter=200)
        if rel_valid and rel_total > best_total:
            best_total = rel_total
            best_valid = True
            best_xy = rel_xy
            best_r = rel_r

    # Phase 2: Simulated annealing from best greedy result
    if best_valid and best_xy is not None:
        if time.time() < deadline - 20:
            sa_xy, sa_r, sa_total, sa_valid = _simulated_annealing(
                best_xy, best_r, deadline - 15, n_iter=3000)
            if sa_valid and sa_total > best_total:
                best_total = sa_total
                best_valid = True
                best_xy = sa_xy
                best_r = sa_r

    # Phase 3: Gradient polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 15:
            polished_xy, polished_r = _gradient_polish(best_xy, best_r, deadline - 10, max_iter=300)
            total, valid = _eval_packing(polished_xy, polished_r)
            if valid and total > best_total:
                best_total = total
                best_xy = polished_xy
                best_r = polished_r

    # Phase 4: Another SA round from polished result
    if best_valid and best_xy is not None:
        if time.time() < deadline - 10:
            sa2_xy, sa2_r, sa2_total, sa2_valid = _simulated_annealing(
                best_xy, best_r, deadline - 5, n_iter=2000)
            if sa2_valid and sa2_total > best_total:
                best_total = sa2_total
                best_valid = True
                best_xy = sa2_xy
                best_r = sa2_r

    # Phase 5: Final gradient polish
    if best_valid and best_xy is not None:
        if time.time() < deadline - 5:
            final_xy, final_r = _gradient_polish(best_xy, best_r, deadline - 2, max_iter=200)
            total, valid = _eval_packing(final_xy, final_r)
            if valid and total > best_total:
                best_total = total
                best_xy = final_xy
                best_r = final_r

    # Fallback
    if best_xy is None or not best_valid:
        side = int(np.ceil(np.sqrt(N)))
        xy = np.zeros((N, 2))
        r = np.zeros(N)
        for i in range(N):
            gx = i % side
            gy = i // side
            xy[i, 0] = (gx + 0.5) / side
            xy[i, 1] = (gy + 0.5) / side
            r[i] = 0.4 / side
        best_xy = xy
        best_r = r

    # Final validation and clipping
    n = len(best_xy)
    result = np.zeros((n, 3))
    for i in range(n):
        x, y, ri = best_xy[i, 0], best_xy[i, 1], best_r[i]
        ri = min(ri, x, 1.0 - x, y, 1.0 - y)
        ri = max(ri, 0.001)
        result[i, 2] = ri
        result[i, 0] = np.clip(x, ri, 1.0 - ri)
        result[i, 1] = np.clip(y, ri, 1.0 - ri)

    final_xy = result[:, :2]
    final_r = result[:, 2]
    total, valid = _eval_packing(final_xy, final_r)
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = 0.4 / side

    return result
# EVOLVE-BLOCK-END
