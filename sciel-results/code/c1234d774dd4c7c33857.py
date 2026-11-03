"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, linprog

DEADLINE = 110.0  # seconds, safety margin before 120s kill


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


def _objective(params, W, H):
    """Minimize negative sum of radii + penalties for overlaps and boundary violations."""
    circles = params.reshape(N, 3)
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Penalty for negative radii
    penalty = 100.0 * np.sum(np.maximum(0, -r))

    # Penalty for going outside rectangle
    penalty += 100.0 * np.sum(np.maximum(0, -x + r))
    penalty += 100.0 * np.sum(np.maximum(0, -y + r))
    penalty += 100.0 * np.sum(np.maximum(0, x - W + r))
    penalty += 100.0 * np.sum(np.maximum(0, y - H + r))

    # Penalty for overlaps
    for i in range(N):
        for j in range(i + 1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            overlap = r[i] + r[j] - dist
            if overlap > 0:
                penalty += 10.0 * overlap * overlap

    # Also penalize if perimeter > 4
    p = 2.0 * (W + H)
    if p > 4.0:
        penalty += 1000.0 * (p - 4.0) ** 2

    return -np.sum(r) + penalty


def _local_optimize(circles, W, H, max_iter=5000):
    """Use Nelder-Mead to locally optimize the circle positions and radii."""
    params = circles.flatten()
    result = minimize(
        _objective, params, args=(W, H),
        method='Nelder-Mead',
        options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
    )
    return result.x.reshape(N, 3)


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
    """Create a random initial configuration with small circles."""
    circles = []
    for _ in range(N):
        r = rng.uniform(0.01, min(W, H) / 6)
        x = rng.uniform(r, W - r)
        y = rng.uniform(r, H - r)
        circles.append([x, y, r])
    return np.array(circles)


def _successive_convex_relaxation(circles, W, H, max_sweeps=50, tol=1e-6):
    """
    Successive Convex Relaxation for circle packing.
    At each sweep, linearize the non-overlap constraints around the current
    positions/radii and solve a QP-like problem via penalty method + Nelder-Mead
    on a smoothed surrogate that uses first-order Taylor approximations of
    the distance constraints.
    """
    n = len(circles)
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()

    best_val = -np.sum(r)
    best_x, best_y, best_r = x.copy(), y.copy(), r.copy()

    for sweep in range(max_sweeps):
        # Build linearized constraints: for each pair (i,j),
        #   dist(i,j) >= r_i + r_j
        # Linearize dist around current (x,y):
        #   dist_ij ≈ dist0_ij + (x_i - x_j - dx0_ij)*(x_i - x_j - dx0_ij)/dist0 + ...
        # Actually use first-order: dist ≈ dist0 + (grad_dist)·(delta)
        # grad_dist w.r.t. x_i = (x_i - x_j)/dist, w.r.t. x_j = -(x_i - x_j)/dist

        # We'll use a smooth penalty that approximates the non-overlap via
        # a log-sum-exp style soft constraint, then do a local Nelder-Mead step.
        # But to be more "SCQP-like", we construct a linearized objective.

        # Simpler approach: use the current configuration, compute all pairwise
        # distances and gradients, then solve a small QP via scipy.
        # Since we don't have a direct QP solver with easy interface for this,
        # we use a penalty-based approach with the linearized constraints as
        # soft terms.

        # Objective: maximize sum(r_i) = minimize -sum(r_i)
        # Linearized non-overlap: for pair (i,j),
        #   g_ij = dist0_ij + sum_k grad_k · (pos_k - pos0_k) - (r_i + r_j) >= 0
        # Penalty: sum max(0, -g_ij)^2

        # Build the penalty function using linearization
        def lin_objective(params):
            p = params.reshape(n, 3)
            px = p[:, 0]
            py = p[:, 1]
            pr = p[:, 2]

            obj = -np.sum(pr)

            # Boundary penalties
            obj += 100.0 * np.sum(np.maximum(0, -px + pr))
            obj += 100.0 * np.sum(np.maximum(0, -py + pr))
            obj += 100.0 * np.sum(np.maximum(0, px - W + pr))
            obj += 100.0 * np.sum(np.maximum(0, py - H + pr))
            obj += 100.0 * np.sum(np.maximum(0, -pr))

            # Linearized non-overlap penalties
            for i in range(n):
                for j in range(i + 1, n):
                    dx0 = x[i] - x[j]
                    dy0 = y[i] - y[j]
                    dist0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
                    if dist0 < 1e-10:
                        dist0 = 1e-10
                    gx_i = dx0 / dist0
                    gy_i = dy0 / dist0
                    gx_j = -dx0 / dist0
                    gy_j = -dy0 / dist0

                    # Linearized distance
                    lin_dist = dist0 + gx_i * (px[i] - x[i]) + gy_i * (py[i] - y[i]) \
                                 + gx_j * (px[j] - x[j]) + gy_j * (py[j] - y[j])
                    violation = (pr[i] + pr[j]) - lin_dist
                    if violation > 0:
                        obj += 10.0 * violation * violation

            return obj

        params = np.column_stack([x, y, r]).flatten()
        result = minimize(
            lin_objective, params,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        new_params = result.x.reshape(n, 3)
        new_x = new_params[:, 0]
        new_y = new_params[:, 1]
        new_r = new_params[:, 2]

        # Clip to valid range
        new_x = np.clip(new_x, new_r, W - new_r)
        new_y = np.clip(new_y, new_r, H - new_r)
        new_r = np.clip(new_r, 0.001, min(W, H) / 2)

        new_val = -np.sum(new_r)
        if new_val < best_val - 1e-8:
            best_val = new_val
            best_x, best_y, best_r = new_x.copy(), new_y.copy(), new_r.copy()

        # Check convergence
        if np.max(np.abs(new_x - x)) < tol and np.max(np.abs(new_y - y)) < tol and np.max(np.abs(new_r - r)) < tol:
            break

        x, y, r = new_x, new_y, new_r

    return np.column_stack([best_x, best_y, best_r]), -best_val


def _best_known(W, H):
    """Return the best configuration found so far for given W, H."""
    # Start with grid
    circles = _grid_initialization(W, H)
    best_val = -np.sum(circles[:, 2])
    if _check_validity(circles, W, H):
        best_circles = circles.copy()
    else:
        best_circles = circles.copy()

    # Try local optimization
    circles_opt = _local_optimize(circles, W, H, max_iter=3000)
    # Clip to valid range
    circles_opt[:, 0] = np.clip(circles_opt[:, 0], circles_opt[:, 2], W - circles_opt[:, 2])
    circles_opt[:, 1] = np.clip(circles_opt[:, 1], circles_opt[:, 2], H - circles_opt[:, 2])
    circles_opt[:, 2] = np.clip(circles_opt[:, 2], 0.001, min(W, H) / 2)

    val = -np.sum(circles_opt[:, 2])
    if _check_validity(circles_opt, W, H) and val < best_val:
        best_val = val
        best_circles = circles_opt.copy()

    # Try successive convex relaxation
    if time.time() < DEADLINE - 15:
        sc_circles, sc_score = _successive_convex_relaxation(circles_opt, W, H, max_sweeps=30)
        sc_circles[:, 0] = np.clip(sc_circles[:, 0], sc_circles[:, 2], W - sc_circles[:, 2])
        sc_circles[:, 1] = np.clip(sc_circles[:, 1], sc_circles[:, 2], H - sc_circles[:, 2])
        sc_circles[:, 2] = np.clip(sc_circles[:, 2], 0.001, min(W, H) / 2)
        if _check_validity(sc_circles, W, H) and sc_score > -best_val:
            best_val = -sc_score
            best_circles = sc_circles.copy()

    return best_circles, -best_val


def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None

    # Try different aspect ratios
    # Perimeter = 2(W+H) = 4, so W+H = 2
    # Aspect ratio a = W/H, so W = a*H, H(a+1) = 2, H = 2/(a+1), W = 2a/(a+1)
    aspect_ratios = [0.5, 0.7, 1.0, 1.2, 1.5, 2.0, 3.0]

    rng = np.random.default_rng(42)

    for ar in aspect_ratios:
        if time.time() - start_time > DEADLINE - 10:
            break
        W = 2.0 * ar / (ar + 1.0)
        H = 2.0 / (ar + 1.0)

        # Grid initialization + local optimization + SC relaxation
        circles, score = _best_known(W, H)
        if score > best_score:
            best_score = score
            best_result = circles.copy()

        # Try a few random restarts with local optimization
        n_restarts = 3
        for _ in range(n_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            circ_rand = _random_initialization(W, H, rng)
            circ_opt = _local_optimize(circ_rand, W, H, max_iter=2000)
            circ_opt[:, 0] = np.clip(circ_opt[:, 0], circ_opt[:, 2], W - circ_opt[:, 2])
            circ_opt[:, 1] = np.clip(circ_opt[:, 1], circ_opt[:, 2], H - circ_opt[:, 2])
            circ_opt[:, 2] = np.clip(circ_opt[:, 2], 0.001, min(W, H) / 2)

            if _check_validity(circ_opt, W, H):
                s = np.sum(circ_opt[:, 2])
                if s > best_score:
                    best_score = s
                    best_result = circ_opt.copy()

    # Final fallback: simple grid in square
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    # Ensure the result is valid; if not, fall back to a safe grid
    if not _check_validity(best_result, 1.0, 1.0):
        # Try to find W, H for the best result
        W = 2.0 / 2.0  # default square
        H = 2.0 / 2.0
        # Check with the aspect ratio that was used
        # For simplicity, just use a safe grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])

    return best_result
# EVOLVE-BLOCK-END
