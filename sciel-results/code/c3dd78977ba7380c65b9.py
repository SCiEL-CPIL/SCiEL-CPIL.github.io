"""Circle packing in perimeter-4 rectangle; force-directed + projected Gauss-Seidel."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 21
DEADLINE = time.time() + 110.0


def _is_valid(circles, tol=1e-6):
    if circles is None:
        return False
    circles = np.asarray(circles)
    if circles.shape[0] != N:
        return False
    if not np.all(np.isfinite(circles)):
        return False
    r = circles[:, 2]
    if np.any(r < -tol):
        return False
    x = circles[:, 0]
    y = circles[:, 1]
    for i in range(N):
        if x[i] < -r[i] - tol or y[i] < -r[i] - tol:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return False
    return True


def _score(circles, tol=1e-6):
    if circles is None:
        return -1e9
    circles = np.asarray(circles)
    if circles.shape[0] != N or not np.all(np.isfinite(circles)):
        return -1e9
    r = circles[:, 2]
    if np.any(r < -tol):
        return -1e9
    x = circles[:, 0]
    y = circles[:, 1]
    for i in range(N):
        if x[i] < -r[i] - tol or y[i] < -r[i] - tol:
            return -1e9
    for i in range(N):
        for j in range(i + 1, N):
            d = math.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                return -1e9
    return float(np.sum(r))


def _project(circles):
    """Projected Gauss-Seidel: enforce non-overlap and boundary at each step.
    Returns (circles, violated)."""
    c = circles.copy()
    r = c[:, 2]
    x = c[:, 0]
    y = c[:, 1]
    violated = False
    for i in range(N):
        # boundary
        if x[i] < -r[i]:
            x[i] = -r[i]
            violated = True
        if y[i] < -r[i]:
            y[i] = -r[i]
            violated = True
        # pairwise with already-placed (j < i)
        for j in range(i):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = math.hypot(dx, dy)
            req = r[i] + r[j]
            if d < req - 1e-12:
                if d < 1e-12:
                    dx, dy = 1e-4, 1e-4
                    d = math.hypot(dx, dy)
                ux, uy = dx / d, dy / d
                # push i out (j is already placed)
                x[i] = x[j] + ux * req
                y[i] = y[j] + uy * req
                violated = True
    c[:, 0] = x
    c[:, 1] = y
    return c, violated


def _force_step(circles, r, k):
    """One force-directed relaxation: repulsion + gravity toward center.
    Returns (circles, energy)."""
    c = circles.copy()
    x = c[:, 0].copy()
    y = c[:, 1].copy()
    cx = 0.5
    cy = 0.5
    fx = np.zeros(N)
    fy = np.zeros(N)
    energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = math.hypot(dx, dy)
            if d < 1e-9:
                dx, dy = 1e-5, 1e-5
                d = math.hypot(dx, dy)
            if d < r[i] + r[j]:
                # repulsive force proportional to overlap
                overlap = r[i] + r[j] - d
                f = k * overlap
                ux, uy = dx / d, dy / d
                fx[i] += f * ux
                fy[i] += f * uy
                fx[j] -= f * ux
                fy[j] -= f * uy
                energy += 0.5 * f * overlap
        # gravity toward center
        fx[i] += 0.5 * (cx - x[i])
        fy[i] += 0.5 * (cy - y[i])
    # update with damping
    for i in range(N):
        x[i] += 0.5 * fx[i]
        y[i] += 0.5 * fy[i]
    c[:, 0] = x
    c[:, 1] = y
    return c, energy


def _solve_for_radii(r, x0, y0, max_iter=2000, k0=10.0, tol=1e-9):
    """Solve for positions given fixed radii using force-directed + projection.
    Returns (circles, converged)."""
    c = np.column_stack([x0.copy(), y0.copy(), r.copy()])
    k = k0
    prev_energy = np.inf
    best_c = c.copy()
    best_energy = np.inf
    for it in range(max_iter):
        if time.time() > DEADLINE - 2:
            break
        c, energy = _force_step(c, r, k)
        c, violated = _project(c)
        if energy < best_energy:
            best_energy = energy
            best_c = c.copy()
        # anneal
        k *= 0.995
        if energy < tol:
            return best_c, True
        if abs(prev_energy - energy) < 1e-12 and it > 50:
            # stuck, restart with small perturbation
            c = best_c.copy()
            c[:, 0] += 0.01 * (np.random.rand(N) - 0.5)
            c[:, 1] += 0.01 * (np.random.rand(N) - 0.5)
        prev_energy = energy
    return best_c, False


def _refine_with_sciipy(circles, max_iter=500):
    """Final high-precision polish using scipy minimize on a smooth objective.
    Objective: minimize sum of squared violations (soft) to find feasible point,
    then maximize sum of radii."""
    c0 = np.asarray(circles).copy()
    n = N

    def objective(v):
        x = v[:n]
        y = v[n:2*n]
        r = v[2*n:3*n]
        # penalty for boundary violations
        pen = 0.0
        for i in range(n):
            if x[i] < -r[i]:
                pen += (x[i] + r[i])**2
            if y[i] < -r[i]:
                pen += (y[i] + r[i])**2
        # penalty for overlap
        for i in range(n):
            for j in range(i+1, n):
                d2 = (x[i]-x[j])**2 + (y[i]-y[j])**2
                req = r[i] + r[j]
                d = math.sqrt(max(d2, 1e-12))
                if d < req:
                    pen += (req - d)**2
        # maximize sum of radii -> minimize -sum(r)
        return pen - 0.001 * np.sum(r)

    v0 = c0.flatten()
    res = minimize(objective, v0, method='L-BFGS-B', options={'maxiter': max_iter, 'ftol': 1e-12})
    if res.success or _score(res.x.reshape(n, 3)) > _score(c0):
        return res.x.reshape(n, 3)
    return c0


def construct():
    t0 = time.time()
    np.random.seed(42)

    # Initial radii: equal, then optimize
    # Start with a grid-like placement
    side = int(math.ceil(math.sqrt(N)))
    x0 = np.zeros(N)
    y0 = np.zeros(N)
    r0 = np.zeros(N)
    for idx in range(N):
        i = idx % side
        j = idx // side
        x0[idx] = (i + 0.5) / side
        y0[idx] = (j + 0.5) / side
        r0[idx] = 0.5 / side - 0.01

    # Try multiple initial configurations and radii
    best_circles = None
    best_score = -1e9

    configs = []
    # Config 1: equal radii from grid
    configs.append((x0.copy(), y0.copy(), r0.copy()))

    # Config 2: slightly varied radii
    r_var = r0.copy() * (1 + 0.1 * (np.random.rand(N) - 0.5))
    configs.append((x0.copy(), y0.copy(), r_var.copy()))

    # Config 3: random perturbation
    x_rand = x0 + 0.05 * (np.random.rand(N) - 0.5)
    y_rand = y0 + 0.05 * (np.random.rand(N) - 0.5)
    configs.append((x_rand, y_rand, r0.copy()))

    for cfg in configs:
        if time.time() > DEADLINE - 10:
            break
        x_c, y_c, r_c = cfg
        # Ensure initial feasibility via projection
        c_init = np.column_stack([x_c, y_c, r_c])
        c_init, _ = _project(c_init)

        # Solve
        c_sol, converged = _solve_for_radii(r_c, c_init[:, 0], c_init[:, 1], max_iter=1500)
        if _is_valid(c_sol):
            s = _score(c_sol)
            if s > best_score:
                best_score = s
                best_circles = c_sol.copy()

    if best_circles is None:
        # Fallback: return the seed-like grid
        side = int(math.ceil(math.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Try to improve radii: iteratively increase smallest radii and re-solve
    improved = best_circles.copy()
    for _ in range(5):
        if time.time() > DEADLINE - 15:
            break
        # Find the smallest radius and try to increase it
        min_idx = np.argmin(improved[:, 2])
        r_new = improved[:, 2].copy()
        r_new[min_idx] *= 1.02  # 2% increase
        # Also try increasing all slightly
        r_all = improved[:, 2].copy() * 1.01
        for r_try in [r_new, r_all]:
            c_try, _ = _solve_for_radii(r_try, improved[:, 0], improved[:, 1], max_iter=1000)
            if _is_valid(c_try):
                s = _score(c_try)
                if s > _score(improved):
                    improved = c_try.copy()

    # Final polish with scipy
    if time.time() < DEADLINE - 5:
        polished = _refine_with_sciipy(improved, max_iter=300)
        if _is_valid(polished) and _score(polished) > _score(improved):
            improved = polished

    # Final validity check
    if not _is_valid(improved):
        # Return best valid found
        if best_circles is not None and _is_valid(best_circles):
            return best_circles
        side = int(math.ceil(math.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    return improved
