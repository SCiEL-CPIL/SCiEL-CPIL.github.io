"""Circle packing in the unit square: maximize sum of 26 disjoint radii."""
import time
import math
import numpy as np

N = 26
_DEADLINE = 150.0
_MARGIN = 2.0
_RNG = np.random.RandomState(12345)


def _objective(params, beta):
    """Penalized objective (maximize). params = [r_0..r_{N-1}, x_0..y_{N-1}]."""
    r = params[:N]
    pos = params[N:].reshape(N, 2)
    x = pos[:, 0]
    y = pos[:, 1]
    score = np.sum(r)

    # Boundary penalty: circle must fit in [0,1]^2
    b = np.maximum(np.maximum(r - x, r - y), np.maximum(r - (1.0 - x), r - (1.0 - y)))
    if np.any(b > 0):
        score -= beta * np.sum(b * b)

    # Pairwise non-overlap penalty
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    need = (r[:, None] + r[None, :]) - d
    np.fill_diagonal(need, 0.0)
    if np.any(need > 0):
        score -= beta * np.sum(need * need)

    return score


def _objective_and_grad(params, beta):
    r = params[:N]
    pos = params[N:].reshape(N, 2)
    x = pos[:, 0]
    y = pos[:, 1]

    grad_r = np.full(N, 1.0)
    grad_pos = np.zeros((N, 2))

    # Boundary
    for i in range(N):
        r_i = r[i]
        xi, yi = x[i], y[i]
        # left: r - x  -> d/d r = 1, d/d x = -1
        v = r_i - xi
        if v > 0:
            grad_r[i] += -2.0 * beta * v
            grad_pos[i, 0] += 2.0 * beta * v
        v = r_i - yi
        if v > 0:
            grad_r[i] += -2.0 * beta * v
            grad_pos[i, 1] += 2.0 * beta * v
        v = r_i - (1.0 - xi)
        if v > 0:
            grad_r[i] += -2.0 * beta * v
            grad_pos[i, 0] += -2.0 * beta * v
        v = r_i - (1.0 - yi)
        if v > 0:
            grad_r[i] += -2.0 * beta * v
            grad_pos[i, 1] += -2.0 * beta * v

    # Pairwise
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    need = (r[:, None] + r[None, :]) - d
    for i in range(N):
        for j in range(i + 1, N):
            v = need[i, j]
            if v > 0:
                # term = beta * v^2 ; dv/dr_i = 1, dv/dr_j = 1, dv/dx_i = -ux, dv/dy_i=-uy
                ux = (x[i] - x[j]) / d[i, j]
                uy = (y[i] - y[j]) / d[i, j]
                g = 2.0 * beta * v
                grad_r[i] += -g
                grad_r[j] += -g
                grad_pos[i, 0] += g * ux
                grad_pos[i, 1] += g * uy
                grad_pos[j, 0] += -g * ux
                grad_pos[j, 1] += -g * uy

    grad = np.concatenate([grad_r, grad_pos.ravel()])
    return _objective(params, beta), grad


def _initial_positions(n):
    """Return (pos, r) for a random-ish initial configuration of n circles."""
    pos = np.zeros((n, 2))
    r = np.zeros(n)
    # mix of grid + jitter + random to give varied topologies
    side = max(1, int(math.ceil(math.sqrt(n))))
    used = 0
    for i in range(side):
        for j in range(side):
            if used >= n:
                break
            pos[used, 0] = (i + 0.5) / side + _RNG.uniform(-0.12, 0.12)
            pos[used, 1] = (j + 0.5) / side + _RNG.uniform(-0.12, 0.12)
            r[used] = 0.45 / side
            used += 1
    # fill any remainder with random points
    while used < n:
        pos[used, 0] = _RNG.uniform(0.1, 0.9)
        pos[used, 1] = _RNG.uniform(0.1, 0.9)
        r[used] = _RNG.uniform(0.02, 0.05)
        used += 1
    # clip centers inside
    pos = np.clip(pos, 0.0, 1.0)
    return pos, r


def _optimize(pos0, r0, start_beta, end_beta, n_sweeps, max_iter, t0):
    from scipy.optimize import minimize
    params = np.concatenate([r0, pos0.ravel()])
    n_params = len(params)
    bounds = [(0.0, 0.5) for _ in range(N)] + [(0.0, 1.0)] * (2 * N)
    # ensure params length matches bounds (N circles)
    # (params already length 3*N)
    best_params = params.copy()
    best_val = _objective(params, end_beta)
    beta = start_beta
    for _ in range(n_sweeps):
        if time.time() > _DEADLINE - _MARGIN:
            break
        res = minimize(
            lambda p, b=beta: _objective(p, b),
            params,
            jac=lambda p, b=beta: _objective_and_grad(p, b)[1],
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "maxfun": max_iter * 4, "ftol": 1e-12, "gtol": 1e-6},
        )
        params = res.x
        val = _objective(params, end_beta)
        if val > best_val:
            best_val = val
            best_params = params.copy()
        # sharpen the penalty
        beta = max(end_beta, beta * 0.35)
    return best_params, best_val


def construct():
    t0 = time.time()
    # Baseline: simple grid of equal circles (safe fallback)
    side = int(math.ceil(math.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
             for i in range(side) for j in range(side)]
    baseline = np.array(cells[:N])
    best = baseline.copy()
    best_score = float(np.sum(baseline[:, 2]))

    start_beta = 1.0
    end_beta = 200000.0
    n_sweeps = 4
    max_iter = 200

    # multiple starts with varied topologies
    n_starts = 6
    for s in range(n_starts):
        if time.time() > _DEADLINE - _MARGIN:
            break
        pos0, r0 = _initial_positions(N)
        # randomize start beta a bit
        sb = start_beta * (1.0 + 0.5 * s)
        params, val = _optimize(pos0, r0, sb, end_beta, n_sweeps, max_iter, t0)
        r = params[:N]
        pos = params[N:].reshape(N, 2)
        # validate: must be inside and non-overlapping (with small tolerance)
        ok = True
        x = pos[:, 0]; y = pos[:, 1]
        if np.any(r < -1e-9) or np.any(x < -1e-9) or np.any(y < -1e-9) or \
           np.any(x > 1 + 1e-9) or np.any(y > 1 + 1e-9) or \
           np.any(r + x < -1e-9) or np.any(r + y < -1e-9) or \
           np.any(r - (1 - x) < -1e-9) or np.any(r - (1 - y) < -1e-9):
            ok = False
        if ok:
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            need = (r[:, None] + r[None, :]) - d
            np.fill_diagonal(need, 0.0)
            if np.any(need > 1e-7):
                ok = False
        if ok and val > best_score:
            best_score = val
            best = np.column_stack([x, y, r])
        elif val > best_score:
            # even if slightly infeasible, keep best feasible baseline; skip
            pass

    # Final: if best is still the baseline, try one more polish on best itself
    if time.time() < _DEADLINE - _MARGIN:
        pos0 = best[:, :2].copy()
        r0 = best[:, 2].copy()
        params, val = _optimize(pos0, r0, end_beta, end_beta, 2, max_iter, t0)
        r = params[:N]
        pos = params[N:].reshape(N, 2)
        x = pos[:, 0]; y = pos[:, 1]
        ok = True
        if np.any(r < -1e-9) or np.any(x < -1e-9) or np.any(y < -1e-9) or \
           np.any(x > 1 + 1e-9) or np.any(y > 1 + 1e-9) or \
           np.any(r + x < -1e-9) or np.any(r + y < -1e-9) or \
           np.any(r - (1 - x) < -1e-9) or np.any(r - (1 - y) < -1e-9):
            ok = False
        if ok:
            d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
            need = (r[:, None] + r[None, :]) - d
            np.fill_diagonal(need, 0.0)
            if np.any(need > 1e-7):
                ok = False
        if ok and val > best_score:
            best_score = val
            best = np.column_stack([x, y, r])

    return best
