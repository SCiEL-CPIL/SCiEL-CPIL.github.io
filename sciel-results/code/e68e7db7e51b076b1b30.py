"""Circle packing in perimeter-4 rectangle: 21 circles, maximize sum of radii."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE = 140.0  # safety margin before 150s kill


def _elapsed(start):
    return time.time() - start


def _is_valid(circles, W, H):
    """Check circles are inside rectangle and non-overlapping."""
    n = len(circles)
    if n != N:
        return False
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r <= 0):
        return False
    if np.any(x - r < -1e-7) or np.any(y - r < -1e-7):
        return False
    if np.any(x + r > W + 1e-7) or np.any(y + r > H + 1e-7):
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-7:
                return False
    return True


def _sum_r(circles):
    return float(np.sum(circles[:, 2]))


def _scl_objective_and_grad(params, W, H, lin_points):
    """
    Successive Convex Relaxation objective + gradient.
    params: (n,3) -> (x, y, r)
    lin_points: previous iterate for linearizing sqrt(x^2+y^2)
    Minimizes -sum(r) subject to linearized non-overlap + boundary constraints.
    """
    n = N
    p = params.reshape(n, 3)
    x, y, r = p[:, 0], p[:, 1], p[:, 2]

    lx, ly, lr = lin_points[:, 0], lin_points[:, 1], lin_points[:, 2]

    # Objective: maximize sum(r) -> minimize -sum(r)
    obj = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0

    # Linearize distance: |p_i - p_j| approx |l_i - l_j| + grad . (p - l)
    # For each pair, linearized distance constraint:
    #   (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # Linearized: 2*(lx_i-lx_j)*(x_i-x_j) + 2*(ly_i-ly_j)*(y_i-y_j)
    #              + (lx_i-lx_j)^2 + (ly_i-ly_j)^2
    #              >= (lr_i+lr_j)^2 + 2*(lr_i+lr_j)*(r_i+r_j - lr_i-lr_j)
    # Simplify: we use the tangent plane at the linearization point.
    #
    # Let d_ij = (lx_i - lx_j, ly_i - ly_j), |d| = dist_l
    # Linearized dist = dist_l + (d . (p_i - l_i) - d . (p_j - l_j)) / dist_l
    #                = dist_l + (d_x*(x_i-x_j - lx_i+lx_j) + d_y*(y_i-y_j - ly_i+ly_j)) / dist_l
    # Constraint: linearized_dist >= r_i + r_j
    # i.e., dist_l + (d_x*(dx - d_x) + d_y*(dy - d_y))/dist_l - r_i - r_j >= 0
    # where dx = x_i-x_j, dy = y_i-y_j
    # = (d_x*dx + d_y*dy)/dist_l - r_i - r_j >= 0
    # = cos_theta * dx + sin_theta * dy - r_i - r_j >= 0
    # where cos_theta = d_x/dist_l, sin_theta = d_y/dist_l

    # We'll add these as soft constraints in the objective (penalty) since
    # SLSQP handles a limited number of constraints efficiently.
    # Actually, let's use SLSQP with explicit inequality constraints.
    # But for the objective+grad interface, we'll use a penalty approach
    # that is exact at the solution.

    # Better: use SLSQP with constraint functions directly.
    # For now, return just the objective and let SLSQP handle constraints.
    # We'll define constraints separately.

    return obj, grad


def _solve_scl(W, H, x0, max_sweeps=30, time_budget=20.0):
    """
    Successive Convex Relaxation: iteratively solve convex subproblems
    with linearized non-overlap constraints using SLSQP.
    """
    start = time.time()
    n = N
    p = x0.copy()  # (n, 3)
    best_p = p.copy()
    best_score = _sum_r(p)

    r_max = min(W, H) / 2.0

    for sweep in range(max_sweeps):
        if _elapsed(start) > time_budget - 2.0:
            break

        # Linearization point = current solution
        lin = p.copy()

        # Define objective
        def obj(params):
            r = params.reshape(n, 3)[:, 2]
            return -np.sum(r)

        # Define gradient
        def obj_grad(params):
            g = np.zeros_like(params)
            g[2::3] = -1.0
            return g

        # Define inequality constraints: g(params) >= 0
        # 1. Boundary: x_i - r_i >= 0, y_i - r_i >= 0, W - x_i - r_i >= 0, H - y_i - r_i >= 0
        # 2. r_i >= 0
        # 3. Non-overlap (linearized): for each pair (i,j):
        #    cos_ij * (x_i - x_j) + sin_ij * (y_i - y_j) - r_i - r_j >= 0

        constraints = []

        # Boundary constraints
        for i in range(n):
            # x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i: params[3*i] - params[3*i+2],
                'jac': lambda params, i=i: np.array([1.0, 0.0, -1.0]).repeat(1) * np.eye(1,1) if False else _bc_jac(params, i, 0)
            })
            # y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i: params[3*i+1] - params[3*i+2],
                'jac': lambda params, i=i: _bc_jac(params, i, 1)
            })
            # W - x_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i: W - params[3*i] - params[3*i+2],
                'jac': lambda params, i=i: _bc_jac(params, i, 2)
            })
            # H - y_i - r_i >= 0
            constraints.append({
                'type': 'ineq',
                'fun': lambda params, i=i: H - params[3*i+1] - params[3*i+2],
                'jac': lambda params, i=i: _bc_jac(params, i, 3)
            })

        # Non-overlap constraints (linearized)
        for i in range(n):
            for j in range(i + 1, n):
                dx_l = lin[i, 0] - lin[j, 0]
                dy_l = lin[i, 1] - lin[j, 1]
                dist_l = np.sqrt(dx_l * dx_l + dy_l * dy_l)
                if dist_l < 1e-10:
                    dist_l = 1e-10
                    dx_l = 1e-10
                    dy_l = 0.0
                cos_t = dx_l / dist_l
                sin_t = dy_l / dist_l

                constraints.append({
                    'type': 'ineq',
                    'fun': lambda params, i=i, j=j, cos_t=cos_t, sin_t=sin_t:
                        cos_t * (params[3*i] - params[3*j]) +
                        sin_t * (params[3*i+1] - params[3*j+1]) -
                        params[3*i+2] - params[3*j+2],
                    'jac': lambda params, i=i, j=j, cos_t=cos_t, sin_t=sin_t: _nc_jac(params, i, j, cos_t, sin_t)
                })

        # Bounds
        bounds = []
        for i in range(n):
            bounds.append((1e-8, W - 1e-8))   # x
            bounds.append((1e-8, H - 1e-8))   # y
            bounds.append((1e-8, r_max))      # r

        params0 = p.flatten()
        try:
            res = minimize(
                obj, params0, jac=obj_grad,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-12, 'disp': False}
            )
        except Exception:
            break

        new_p = res.x.reshape(n, 3)
        # Clamp to feasible region
        new_p[:, 0] = np.clip(new_p[:, 0], new_p[:, 2], W - new_p[:, 2])
        new_p[:, 1] = np.clip(new_p[:, 1], new_p[:, 2], H - new_p[:, 2])
        new_p[:, 2] = np.clip(new_p[:, 2], 1e-8, r_max)

        new_score = _sum_r(new_p)
        if new_score > best_score - 1e-10:
            best_score = new_score
            best_p = new_p.copy()

        # Check convergence
        if np.max(np.abs(new_p - p)) < 1e-8:
            break
        p = new_p

    return best_p, best_score


def _bc_jac(params, i, which):
    """Jacobian for boundary constraint. which: 0=x-r, 1=y-r, 2=W-x-r, 3=H-y-r"""
    g = np.zeros(len(params))
    if which == 0:  # x_i - r_i
        g[3*i] = 1.0
        g[3*i+2] = -1.0
    elif which == 1:  # y_i - r_i
        g[3*i+1] = 1.0
        g[3*i+2] = -1.0
    elif which == 2:  # W - x_i - r_i
        g[3*i] = -1.0
        g[3*i+2] = -1.0
    elif which == 3:  # H - y_i - r_i
        g[3*i+1] = -1.0
        g[3*i+2] = -1.0
    return g


def _nc_jac(params, i, j, cos_t, sin_t):
    """Jacobian for non-overlap constraint: cos*(xi-xj) + sin*(yi-yj) - ri - rj"""
    g = np.zeros(len(params))
    g[3*i] = cos_t
    g[3*i+1] = sin_t
    g[3*i+2] = -1.0
    g[3*j] = -cos_t
    g[3*j+1] = -sin_t
    g[3*j+2] = -1.0
    return g


def _momentum_repulsion(circles, W, H, n_iters=200, dt=0.01):
    """
    Momentum-based repulsion to resolve overlaps and push circles apart.
    Returns improved circles.
    """
    n = N
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    vx = np.zeros(n)
    vy = np.zeros(n)
    damping = 0.95

    for it in range(n_iters):
        fx = np.zeros(n)
        fy = np.zeros(n)

        # Boundary forces
        for i in range(n):
            if x[i] < r[i]:
                fx[i] += 100.0 * (r[i] - x[i])
            if x[i] > W - r[i]:
                fx[i] -= 100.0 * (x[i] - (W - r[i]))
            if y[i] < r[i]:
                fy[i] += 100.0 * (r[i] - y[i])
            if y[i] > H - r[i]:
                fy[i] -= 100.0 * (y[i] - (H - r[i]))

        # Overlap forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = r[i] + r[j]
                if dist < min_dist and dist > 1e-12:
                    force = 50.0 * (min_dist - dist) / dist
                    fx[i] += force * dx
                    fy[i] += force * dy
                    fx[j] -= force * dx
                    fy[j] -= force * dy
                elif dist <= 1e-12:
                    # Random nudge
                    angle = np.random.uniform(0, 2 * np.pi)
                    fx[i] += np.cos(angle)
                    fy[i] += np.sin(angle)
                    fx[j] -= np.cos(angle)
                    fy[j] -= np.sin(angle)

        # Update velocities with momentum
        vx = damping * vx + dt * fx
        vy = damping * vy + dt * fy

        # Update positions
        x += vx
        y += vy

        # Hard clamp
        for i in range(n):
            x[i] = np.clip(x[i], r[i], W - r[i])
            y[i] = np.clip(y[i], r[i], H - r[i])

    return np.column_stack([x, y, r])


def _grow_radii(circles, W, H, n_steps=50, growth=0.005):
    """
    Iteratively grow all radii and resolve collisions.
    Returns best valid configuration found.
    """
    best = circles.copy()
    best_score = _sum_r(circles)

    for step in range(n_steps):
        r_new = circles[:, 2] + growth
        # Check if growth is feasible (rough check)
        if np.any(r_new > min(W, H) / 2.0):
            break

        circles_try = circles.copy()
        circles_try[:, 2] = r_new

        # Resolve overlaps via momentum repulsion (short)
        circles_try = _momentum_repulsion(circles_try, W, H, n_iters=50, dt=0.005)

        # Clamp
        circles_try[:, 0] = np.clip(circles_try[:, 0], circles_try[:, 2], W - circles_try[:, 2])
        circles_try[:, 1] = np.clip(circles_try[:, 1], circles_try[:, 2], H - circles_try[:, 2])
        circles_try[:, 2] = np.clip(circles_try[:, 2], 1e-8, min(W, H) / 2.0)

        if _is_valid(circles_try, W, H):
            score = _sum_r(circles_try)
            if score > best_score:
                best_score = score
                best = circles_try.copy()
            circles = circles_try.copy()
        else:
            # Try with smaller growth
            circles_try2 = circles.copy()
            circles_try2[:, 2] = circles[:, 2] + growth * 0.5
            circles_try2 = _momentum_repulsion(circles_try2, W, H, n_iters=30, dt=0.003)
            circles_try2[:, 0] = np.clip(circles_try2[:, 0], circles_try2[:, 2], W - circles_try2[:, 2])
            circles_try2[:, 1] = np.clip(circles_try2[:, 1], circles_try2[:, 2], H - circles_try2[:, 2])
            if _is_valid(circles_try2, W, H):
                score = _sum_r(circles_try2)
                if score > best_score:
                    best_score = score
                    best = circles_try2.copy()
                circles = circles_try2.copy()
            break

    return best, best_score


def _grid_init(W, H):
    """Create a 5x5 grid with one extra circle (21 = 4*5+1)."""
    # Use 5 columns, 4 rows + 1 extra
    cols, rows = 5, 4
    circles = []
    for i in range(cols):
        for j in range(rows):
            cx = (i + 0.5) * W / cols
            cy = (j + 0.5) * H / rows
            r = min(W / cols, H / rows) * 0.45
            circles.append([cx, cy, r])
    # Extra circle in center
    circles.append([W / 2.0, H / 2.0, min(W, H) * 0.08])
    return np.array(circles[:N])


def _random_init(W, H, rng, n=N):
    """Random initialization with small circles."""
    circles = []
    for _ in range(n):
        r = rng.uniform(0.02, 0.12)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _pack_for_aspect(W, H, time_budget=15.0, seed=42):
    """
    Pack circles in a rectangle of size W x H.
    Uses SCL + momentum repulsion + radius growth.
    Returns (circles, score).
    """
    start = time.time()
    rng = np.random.default_rng(seed)
    best_circles = None
    best_score = 0.0

    # Phase 1: Grid initialization + SCL
    if _elapsed(start) < time_budget - 3.0:
        circ = _grid_init(W, H)
        # Separate initial overlaps
        circ = _momentum_repulsion(circ, W, H, n_iters=100, dt=0.01)
        circ[:, 0] = np.clip(circ[:, 0], circ[:, 2], W - circ[:, 2])
        circ[:, 1] = np.clip(circ[:, 1], circ[:, 2], H - circ[:, 2])

        # SCL optimization
        scl_circ, scl_score = _solve_scl(W, H, circ, max_sweeps=20,
                                          time_budget=min(8.0, time_budget - 3.0))
        if _is_valid(scl_circ, W, H):
            s = _sum_r(scl_circ)
            if s > best_score:
                best_score = s
                best_circles = scl_circ.copy()
        elif _is_valid(circ, W, H):
            s = _sum_r(circ)
            if s > best_score:
                best_score = s
                best_circles = circ.copy()

    # Phase 2: Radius growth from best
    if best_circles is not None and _elapsed(start) < time_budget - 2.0:
        grown, grow_score = _grow_radii(best_circles, W, H, n_steps=40, growth=0.003)
        if grow_score > best_score:
            best_score = grow_score
            best_circles = grown.copy()

    # Phase 3: SCL on grown result
    if best_circles is not None and _elapsed(start) < time_budget - 2.0:
        scl2_circ, scl2_score = _solve_scl(W, H, best_circles, max_sweeps=15,
                                            time_budget=min(5.0, time_budget - 2.0))
        if _is_valid(scl2_circ, W, H):
            s = _sum_r(scl2_circ)
            if s > best_score:
                best_score = s
                best_circles = scl2_circ.copy()

    # Phase 4: Random restarts
    remaining = time_budget - _elapsed(start)
    if remaining > 3.0 and best_circles is not None:
        n_restarts = max(1, int(remaining / 4.0))
        for k in range(n_restarts):
            if _elapsed(start) > time_budget - 1.0:
                break
            circ_r = _random_init(W, H, rng, seed=seed + 100 + k)
            circ_r = _momentum_repulsion(circ_r, W, H, n_iters=80, dt=0.01)
            circ_r[:, 0] = np.clip(circ_r[:, 0], circ_r[:, 2], W - circ_r[:, 2])
            circ_r[:, 1] = np.clip(circ_r[:, 1], circ_r[:, 2], H - circ_r[:, 2])

            scl_r, scl_r_score = _solve_scl(W, H, circ_r, max_sweeps=10,
                                             time_budget=min(3.0, time_budget - _elapsed(start) - 1.0))
            if _is_valid(scl_r, W, H):
                s = _sum_r(scl_r)
                if s > best_score:
                    best_score = s
                    best_circles = scl_r.copy()

    # Phase 5: Final polish with SCL
    if best_circles is not None and _elapsed(start) < time_budget - 0.5:
        polished, pol_score = _solve_scl(W, H, best_circles, max_sweeps=10,
                                          time_budget=min(3.0, time_budget - _elapsed(start) - 0.5))
        if _is_valid(polished, W, H):
            s = _sum_r(polished)
            if s > best_score:
                best_score = s
                best_circles = polished.copy()

    if best_circles is None:
        # Fallback: small grid
        best_circles = _grid_init(W, H)
        best_circles = _momentum_repulsion(best_circles, W, H, n_iters=100, dt=0.01)
        best_circles[:, 0] = np.clip(best_circles[:, 0], best_circles[:, 2], W - best_circles[:, 2])
        best_circles[:, 1] = np.clip(best_circles[:, 1], best_circles[:, 2], H - best_circles[:, 2])
        best_score = _sum_r(best_circles)

    return best_circles, best_score


def construct():
    start_time = time.time()
    best_score = 0.0
    best_result = None

    # Aspect ratios to try (W/H ratio)
    # Perimeter = 2(W+H) = 4, so W+H = 2
    # aspect = W/H, so W = 2*ar/(ar+1), H = 2/(ar+1)
    aspect_ratios = [1.0, 0.85, 1.15, 0.7, 1.3, 0.6, 1.5]

    # Sort by priority: square first, then near-square
    priorities = [1.0, 0.85, 1.15, 0.7, 1.3, 0.6, 1.5]

    for ar in priorities:
        elapsed = _elapsed(start_time)
        if elapsed > DEADLINE - 10.0:
            break

        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Allocate time based on aspect ratio priority
        if ar == 1.0:
            tb = min(25.0, DEADLINE - elapsed - 5.0)
        elif ar in (0.85, 1.15):
            tb = min(18.0, DEADLINE - elapsed - 5.0)
        else:
            tb = min(12.0, DEADLINE - elapsed - 5.0)

        if tb < 3.0:
            continue

        circles, score = _pack_for_aspect(W, H, time_budget=tb, seed=int(ar * 1000))
        if score > best_score:
            best_score = score
            best_result = circles.copy()

    # Final: if we have time, do a golden-section search on aspect ratio
    elapsed = _elapsed(start_time)
    if elapsed < DEADLINE - 15.0 and best_result is not None:
        # Refine around the best aspect ratio
        # Find best ar from previous
        best_ar = 1.0
        for ar in priorities:
            W_test = 2.0 * ar / (ar + 1.0)
            H_test = 2.0 / (ar + 1.0)
            # Just use the one that gave best score (we'd need to track this)
            pass

        # Try a few refinements around square
        for ar_ref in [0.95, 1.05, 0.9, 1.1]:
            elapsed = _elapsed(start_time)
            if elapsed > DEADLINE - 8.0:
                break
            W = 2.0 * ar_ref / (ar_ref + 1.0)
            H = 2.0 / (ar_ref + 1.0)
            tb = min(8.0, DEADLINE - elapsed - 3.0)
            if tb < 3.0:
                continue
            circles, score = _pack_for_aspect(W, H, time_budget=tb, seed=int(ar_ref * 10000))
            if score > best_score:
                best_score = score
                best_result = circles.copy()

    # Fallback
    if best_result is None or not _is_valid(best_result, 1.0, 1.0):
        # Simple 5x5 grid in unit square (perimeter 4)
        cols, rows = 5, 5
        cells = []
        for i in range(cols):
            for j in range(rows):
                if len(cells) >= N:
                    break
                cx = (i + 0.5) / cols
                cy = (j + 0.5) / rows
                r = 0.5 / cols * 0.8
                cells.append([cx, cy, r])
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
