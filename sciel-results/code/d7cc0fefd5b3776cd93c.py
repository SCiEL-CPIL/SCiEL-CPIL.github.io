"""Circle packing in unit square: pack 26 disjoint circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 26
DEADLINE = 150.0
SAFETY_MARGIN = 15.0  # seconds to spare


def _elapsed():
    """Return elapsed time since module load (approximate)."""
    return time.time() - _T0


_T0 = time.time()


def _validate_and_score(circles):
    """
    Validate circles: inside [0,1]^2, disjoint interiors.
    Return sum of radii if valid, else -inf.
    circles: (n, 3) array of (x, y, r)
    """
    n = len(circles)
    if n != N:
        return -np.inf
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]

    # Check radii positive
    if np.any(r <= 0):
        return -np.inf

    # Check inside unit square
    if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
        return -np.inf
    if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
        return -np.inf

    # Check disjoint interiors: distance between centers >= r_i + r_j
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - 1e-9:
                return -np.inf

    return float(np.sum(r))


def _logsumexp_penalty(x, y, r, beta=50.0):
    """
    Smooth penalty for overlap constraints using log-sum-exp.
    Constraint: dist(i,j) >= r_i + r_j for all i < j
    Penalty: sum over pairs of logsumexp of (r_i + r_j - dist) with soft max.
    """
    n = len(r)
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            violation = r[i] + r[j] - dist
            # Soft max: use logsumexp of [0, violation] scaled by beta
            # This gives a smooth approximation of max(0, violation)
            if violation > 0:
                penalty += np.log(1.0 + np.exp(beta * violation)) / beta
            else:
                penalty += np.log(1.0 + np.exp(beta * violation)) / beta
    # Also penalize going outside the square
    for i in range(n):
        # x - r >= 0
        v1 = -(x[i] - r[i])
        if v1 > 0:
            penalty += np.log(1.0 + np.exp(beta * v1)) / beta
        else:
            penalty += np.log(1.0 + np.exp(beta * v1)) / beta
        # x + r <= 1
        v2 = x[i] + r[i] - 1.0
        if v2 > 0:
            penalty += np.log(1.0 + np.exp(beta * v2)) / beta
        else:
            penalty += np.log(1.0 + np.exp(beta * v2)) / beta
        # y - r >= 0
        v3 = -(y[i] - r[i])
        if v3 > 0:
            penalty += np.log(1.0 + np.exp(beta * v3)) / beta
        else:
            penalty += np.log(1.0 + np.exp(beta * v3)) / beta
        # y + r <= 1
        v4 = y[i] + r[i] - 1.0
        if v4 > 0:
            penalty += np.log(1.0 + np.exp(beta * v4)) / beta
        else:
            penalty += np.log(1.0 + np.exp(beta * v4)) / beta
    return penalty


def _objective_and_gradient(params):
    """
    Objective: maximize sum(r) - penalty
    params: (3N,) array [x_0..x_{N-1}, y_0..y_{N-1}, r_0..r_{N-1}]
    """
    x = params[:N]
    y = params[N:2 * N]
    r = params[2 * N:]

    obj = np.sum(r)
    grad = np.zeros_like(params)
    grad[2 * N:] = 1.0  # d(sum r)/dr_i = 1

    # Penalty gradient via finite differences would be expensive;
    # Instead use the log-sum-exp form with analytical gradient.
    # For efficiency, compute penalty and its gradient together.
    beta = 50.0
    n = N

    # Pairwise overlap penalty
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy) + 1e-12
            violation = r[i] + r[j] - dist

            # soft_plus(violation) = log(1 + exp(beta * violation)) / beta
            # d/dviolation = exp(beta * violation) / (1 + exp(beta * violation)) = sigmoid(beta*violation)
            # To avoid overflow:
            if violation > 0:
                sig = 1.0 / (1.0 + np.exp(-beta * violation))
                pen_val = (np.log(1.0 + np.exp(beta * violation))) / beta
            else:
                sig = np.exp(beta * violation) / (1.0 + np.exp(beta * violation))
                pen_val = np.log(1.0 + np.exp(beta * violation)) / beta

            obj -= pen_val

            # Gradient of penalty w.r.t. x_i, y_i, r_i, x_j, y_j, r_j
            # d pen_val / d violation = sig
            # d violation / d x_i = -dx/dist
            # d violation / d y_i = -dy/dist
            # d violation / d r_i = 1
            # d violation / d x_j = dx/dist
            # d violation / d y_j = dy/dist
            # d violation / d r_j = 1

            dpen_dviolation = sig
            dviolation_dx_i = -dx / dist
            dviolation_dy_i = -dy / dist
            dviolation_dr_i = 1.0
            dviolation_dx_j = dx / dist
            dviolation_dy_j = dy / dist
            dviolation_dr_j = 1.0

            grad[i] += -dpen_dviolation * dviolation_dx_i
            grad[N + i] += -dpen_dviolation * dviolation_dy_i
            grad[2 * N + i] += -dpen_dviolation * dviolation_dr_i
            grad[j] += -dpen_dviolation * dviolation_dx_j
            grad[N + j] += -dpen_dviolation * dviolation_dy_j
            grad[2 * N + j] += -dpen_dviolation * dviolation_dr_j

    # Boundary penalties
    for i in range(n):
        # x_i - r_i >= 0
        v1 = -(x[i] - r[i])
        if v1 > 0:
            sig1 = 1.0 / (1.0 + np.exp(-beta * v1))
            pen1 = np.log(1.0 + np.exp(beta * v1)) / beta
        else:
            sig1 = np.exp(beta * v1) / (1.0 + np.exp(beta * v1))
            pen1 = np.log(1.0 + np.exp(beta * v1)) / beta
        obj -= pen1
        # d pen1 / d v1 = sig1; d v1 / d x_i = -1; d v1 / d r_i = 1
        grad[i] += -sig1 * (-1.0)
        grad[2 * N + i] += -sig1 * (1.0)

        # x_i + r_i <= 1
        v2 = x[i] + r[i] - 1.0
        if v2 > 0:
            sig2 = 1.0 / (1.0 + np.exp(-beta * v2))
            pen2 = np.log(1.0 + np.exp(beta * v2)) / beta
        else:
            sig2 = np.exp(beta * v2) / (1.0 + np.exp(beta * v2))
            pen2 = np.log(1.0 + np.exp(beta * v2)) / beta
        obj -= pen2
        # d v2 / d x_i = 1; d v2 / d r_i = 1
        grad[i] += -sig2 * 1.0
        grad[2 * N + i] += -sig2 * 1.0

        # y_i - r_i >= 0
        v3 = -(y[i] - r[i])
        if v3 > 0:
            sig3 = 1.0 / (1.0 + np.exp(-beta * v3))
            pen3 = np.log(1.0 + np.exp(beta * v3)) / beta
        else:
            sig3 = np.exp(beta * v3) / (1.0 + np.exp(beta * v3))
            pen3 = np.log(1.0 + np.exp(beta * v3)) / beta
        obj -= pen3
        grad[N + i] += -sig3 * (-1.0)
        grad[2 * N + i] += -sig3 * (1.0)

        # y_i + r_i <= 1
        v4 = y[i] + r[i] - 1.0
        if v4 > 0:
            sig4 = 1.0 / (1.0 + np.exp(-beta * v4))
            pen4 = np.log(1.0 + np.exp(beta * v4)) / beta
        else:
            sig4 = np.exp(beta * v4) / (1.0 + np.exp(beta * v4))
            pen4 = np.log(1.0 + np.exp(beta * v4)) / beta
        obj -= pen4
        grad[N + i] += -sig4 * 1.0
        grad[2 * N + i] += -sig4 * 1.0

    return -obj, -grad  # minimize negative objective


def _polish(params, maxiter=2000):
    """
    Polish a solution using L-BFGS-B with bounds.
    """
    x = params[:N]
    y = params[N:2 * N]
    r = params[2 * N:]

    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0))  # x
    for i in range(N):
        bounds.append((0.0, 1.0))  # y
    for i in range(N):
        bounds.append((1e-6, 0.5))  # r (max radius is 0.5 for a single circle)

    res = minimize(
        _objective_and_gradient,
        params,
        jac=True,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
    )
    return res.x, res.fun


def _polish_slsqp(params, maxiter=1000):
    """
    Polish using SLSQP with explicit constraints.
    """
    n = N
    # Objective: maximize sum(r) => minimize -sum(r)
    def obj(p):
        r = p[2 * n:]
        return -np.sum(r)

    def obj_grad(p):
        g = np.zeros_like(p)
        g[2 * n:] = -1.0
        return g

    # Constraints: dist(i,j) - r_i - r_j >= 0
    cons = []
    for i in range(n):
        for j in range(i + 1, n):
            def con(p, i=i, j=j):
                xi, yi, ri = p[i], p[n + i], p[2 * n + i]
                xj, yj, rj = p[j], p[n + j], p[2 * n + j]
                return np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) - ri - rj

            def con_grad(p, i=i, j=j):
                g = np.zeros_like(p)
                xi, yi, ri = p[i], p[n + i], p[2 * n + i]
                xj, yj, rj = p[j], p[n + j], p[2 * n + j]
                dx = xi - xj
                dy = yi - yj
                dist = np.sqrt(dx * dx + dy * dy) + 1e-12
                g[i] = dx / dist
                g[n + i] = dy / dist
                g[2 * n + i] = -1.0
                g[j] = -dx / dist
                g[n + j] = -dy / dist
                g[2 * n + j] = -1.0
                return g

            cons.append({'type': 'ineq', 'fun': con, 'jac': con_grad})

    # Boundary constraints
    for i in range(n):
        # x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2 * n + i],
                     'jac': lambda p, i=i: _boundary_grad(p, i, 'x_min')})
        # x_i + r_i <= 1 => 1 - x_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[i] - p[2 * n + i],
                     'jac': lambda p, i=i: _boundary_grad(p, i, 'x_max')})
        # y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[n + i] - p[2 * n + i],
                     'jac': lambda p, i=i: _boundary_grad(p, i, 'y_min')})
        # y_i + r_i <= 1 => 1 - y_i - r_i >= 0
        cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[n + i] - p[2 * n + i],
                     'jac': lambda p, i=i: _boundary_grad(p, i, 'y_max')})

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((1e-6, 0.5))

    res = minimize(
        obj,
        params,
        jac=obj_grad,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': maxiter, 'ftol': 1e-12}
    )
    return res.x, res.fun


def _boundary_grad(p, i, which):
    g = np.zeros_like(p)
    if which == 'x_min':
        g[i] = 1.0
        g[2 * N + i] = -1.0
    elif which == 'x_max':
        g[i] = -1.0
        g[2 * N + i] = -1.0
    elif which == 'y_min':
        g[N + i] = 1.0
        g[2 * N + i] = -1.0
    elif which == 'y_max':
        g[N + i] = -1.0
        g[2 * N + i] = -1.0
    return g


def _initial_grid_seed():
    """Grid-based initial seed."""
    side = int(np.ceil(np.sqrt(N)))
    params = np.zeros(3 * N)
    for idx in range(N):
        i = idx % side
        j = idx // side
        params[idx] = (i + 0.5) / side
        params[N + idx] = (j + 0.5) / side
        params[2 * N + idx] = 0.5 / side - 0.01
    return params


def _initial_random_seed(rng):
    """Random initial seed with small radii."""
    params = np.zeros(3 * N)
    for idx in range(N):
        params[idx] = rng.uniform(0.1, 0.9)
        params[N + idx] = rng.uniform(0.1, 0.9)
        params[2 * N + idx] = rng.uniform(0.01, 0.08)
    return params


def _initial_structural_seed(seed_type, rng):
    """Various structural seeds."""
    params = np.zeros(3 * N)
    if seed_type == 0:
        # Hexagonal-like packing
        cols = 5
        rows = 6
        for idx in range(N):
            i = idx % cols
            j = idx // cols
            offset = 0.5 / cols if j % 2 == 1 else 0.0
            params[idx] = (i + 0.5 + offset) / cols
            params[N + idx] = (j + 0.5) / rows
            params[2 * N + idx] = 0.4 / cols
    elif seed_type == 1:
        # Ring + center
        # 1 center, 6 inner ring, 12 outer ring, 7 extra
        cx, cy = 0.5, 0.5
        params[0] = cx
        params[N] = cy
        params[2 * N] = 0.08
        angle_step = 2 * np.pi / 6
        for k in range(6):
            idx = 1 + k
            params[idx] = cx + 0.16 * np.cos(k * angle_step)
            params[N + idx] = cy + 0.16 * np.sin(k * angle_step)
            params[2 * N + idx] = 0.07
        angle_step2 = 2 * np.pi / 12
        for k in range(12):
            idx = 7 + k
            params[idx] = cx + 0.32 * np.cos(k * angle_step2)
            params[N + idx] = cy + 0.32 * np.sin(k * angle_step2)
            params[2 * N + idx] = 0.06
        # Fill remaining 1
        idx = 19
        params[idx] = 0.5
        params[N + idx] = 0.5
        params[2 * N + idx] = 0.03
    elif seed_type == 2:
        # Two clusters
        for idx in range(13):
            i = idx % 4
            j = idx // 4
            params[idx] = 0.25 + (i - 1.5) * 0.12
            params[N + idx] = 0.25 + (j - 1.5) * 0.12
            params[2 * N + idx] = 0.055
        for idx in range(13, 26):
            i = (idx - 13) % 4
            j = (idx - 13) // 4
            params[idx] = 0.75 + (i - 1.5) * 0.12
            params[N + idx] = 0.75 + (j - 1.5) * 0.12
            params[2 * N + idx] = 0.055
    else:
        # Random
        for idx in range(N):
            params[idx] = rng.uniform(0.1, 0.9)
            params[N + idx] = rng.uniform(0.1, 0.9)
            params[2 * N + idx] = rng.uniform(0.02, 0.06)
    return params


def construct():
    global _T0
    _T0 = time.time()

    rng = np.random.default_rng(42)
    best_params = None
    best_score = -np.inf

    # Try multiple initial seeds and polish
    seeds = []
    seeds.append(_initial_grid_seed())
    for st in range(3):
        seeds.append(_initial_structural_seed(st, rng))
    for _ in range(4):
        seeds.append(_initial_random_seed(rng))

    deadline = _T0 + DEADLINE - SAFETY_MARGIN

    # Phase 1: Polish each seed with L-BFGS-B
    for s_idx, seed in enumerate(seeds):
        if time.time() > deadline:
            break
        try:
            polished, _ = _polish(seed, maxiter=1000)
            circles = np.column_stack([polished[:N], polished[N:2 * N], polished[2 * N:]])
            score = _validate_and_score(circles)
            if score > best_score:
                best_score = score
                best_params = polished.copy()
        except Exception:
            pass

    # Phase 2: Differential evolution for global exploration
    if time.time() < deadline:
        bounds_de = []
        for i in range(N):
            bounds_de.append((0.0, 1.0))
        for i in range(N):
            bounds_de.append((0.0, 1.0))
        for i in range(N):
            bounds_de.append((1e-6, 0.5))

        def de_objective(p):
            x = p[:N]
            y = p[N:2 * N]
            r = p[2 * N:]
            # Maximize sum(r) - penalty
            pen = _logsumexp_penalty(x, y, r, beta=30.0)
            return -(np.sum(r)) + pen

        try:
            de_res = differential_evolution(
                de_objective,
                bounds_de,
                maxiter=30,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=True,
                init='sobol',
                update_interval=1,
                mutation=(0.3, 0.8),
                recombination=0.9,
                seed=42,
                workers=-1,
                polish=True
            )
            if de_res.success or de_res.fun < 0:
                circles = np.column_stack([de_res.x[:N], de_res.x[N:2 * N], de_res.x[2 * N:]])
                score = _validate_and_score(circles)
                if score > best_score:
                    best_score = score
                    best_params = de_res.x.copy()
        except Exception:
            pass

    # Phase 3: Multi-restart polish from best and perturbed versions
    if best_params is not None and time.time() < deadline:
        for restart in range(8):
            if time.time() > deadline:
                break
            perturbed = best_params + rng.normal(0, 0.02, size=3 * N)
            # Clamp to bounds
            perturbed[:N] = np.clip(perturbed[:N], 0.0, 1.0)
            perturbed[N:2 * N] = np.clip(perturbed[N:2 * N], 0.0, 1.0)
            perturbed[2 * N:] = np.clip(perturbed[2 * N:], 1e-6, 0.5)
            try:
                polished, _ = _polish(perturbed, maxiter=1500)
                circles = np.column_stack([polished[:N], polished[N:2 * N], polished[2 * N:]])
                score = _validate_and_score(circles)
                if score > best_score:
                    best_score = score
                    best_params = polished.copy()
            except Exception:
                pass

    # Phase 4: SLSQP high-precision polish
    if best_params is not None and time.time() < deadline:
        try:
            polished, _ = _polish_slsqp(best_params, maxiter=500)
            circles = np.column_stack([polished[:N], polished[N:2 * N], polished[2 * N:]])
            score = _validate_and_score(circles)
            if score > best_score:
                best_score = score
                best_params = polished.copy()
        except Exception:
            pass

    # Final validation and return
    if best_params is None:
        # Fallback: trivial grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Ensure validity: if slightly invalid, shrink radii slightly
    x = best_params[:N].copy()
    y = best_params[N:2 * N].copy()
    r = best_params[2 * N:].copy()

    # Check and fix any minor violations
    for _ in range(10):
        valid = True
        for i in range(N):
            if x[i] - r[i] < -1e-10 or x[i] + r[i] > 1 + 1e-10:
                valid = False
            if y[i] - r[i] < -1e-10 or y[i] + r[i] > 1 + 1e-10:
                valid = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r[i] + r[j] - 1e-10:
                    valid = False
        if valid:
            break
        # Shrink radii slightly
        r *= 0.999
        # Adjust positions to be safe
        for i in range(N):
            x[i] = np.clip(x[i], r[i] + 1e-6, 1 - r[i] - 1e-6)
            y[i] = np.clip(y[i], r[i] + 1e-6, 1 - r[i] - 1e-6)

    result = np.column_stack([x, y, r])
    return result
