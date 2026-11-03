import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time

def _compute_radius(c):
    """Exact max radius: min(min_pair_dist/2, min wall clearance)."""
    if c.shape[0] < 2:
        return 0.5 - np.max(np.abs(c))
    wall = 0.5 - np.max(np.abs(c))
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    min_pair = np.min(d) / 2.0
    return min(wall, min_pair)

def _soft_min_objective_and_grad(x, beta):
    """Negative soft-min of all constraints (wall clearances + pair half-distances)."""
    c = x.reshape(N, 3)

    # Wall clearances per coordinate
    wall = 0.5 - np.abs(c)              # (N, 3)
    wall_flat = wall.ravel()            # (3N,)

    # Pairwise half-distances (upper triangle)
    d = cdist(c, c)
    np.fill_diagonal(d, np.inf)
    d_half = d / 2.0
    i, j = np.triu_indices(N, k=1)
    d_half_upper = d_half[i, j]         # (n_pairs,)

    vals = np.concatenate([wall_flat, d_half_upper])
    n_vals = len(vals)

    # Numerically stable log-sum-exp
    m = np.min(vals)
    shifted = vals - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    log_sum_exp = np.log(sum_exp)
    soft_min_val = m - log_sum_exp / beta

    weights = exp_vals / sum_exp        # (n_vals,)

    grad_x = np.zeros_like(c)           # (N, 3)

    # Wall contribution: d(0.5-|c_ik|)/dc_ik = -sign(c_ik)
    ww = weights[:3 * N].reshape(N, 3)
    sign_c = np.sign(c)
    grad_x -= ww * sign_c

    # Pair contribution: d(||ci-cj||/2)/dci = (ci-cj)/(2||ci-cj||)
    pw = weights[3 * N:]                # (n_pairs,)
    ci = c[i]
    cj = c[j]
    diff = ci - cj
    norms = np.linalg.norm(diff, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    grad_ci = diff / (2.0 * norms)
    grad_cj = -grad_ci

    np.add.at(grad_x, i, pw[:, np.newaxis] * grad_ci)
    np.add.at(grad_x, j, pw[:, np.newaxis] * grad_cj)

    f = -soft_min_val
    g = -grad_x.ravel()
    return f, g

def _generate_bcc_lattice(n_points, seed=123):
    """Points on a BCC lattice scaled to [-0.5,0.5]^3, then greedily pick well-spread ones."""
    n_div = 5
    pts = []
    for i in range(n_div + 1):
        for j in range(n_div + 1):
            for k in range(n_div + 1):
                pts.append([i / n_div, j / n_div, k / n_div])
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                pts.append([(i + 0.5) / n_div, (j + 0.5) / n_div, (k + 0.5) / n_div])
    pts = np.array(pts) - 0.5

    rng = np.random.default_rng(seed)
    idx = np.arange(len(pts))
    rng.shuffle(idx)
    pts = pts[idx]

    selected = [pts[0]]
    for p in pts[1:]:
        if len(selected) >= n_points:
            break
        dmin = np.min(np.linalg.norm(np.array(selected) - p, axis=1))
        if dmin > 0.08:
            selected.append(p)
    while len(selected) < n_points:
        selected.append(rng.uniform(-0.45, 0.45, size=3))
    return np.array(selected[:n_points])

def _grid_initial(n_points):
    m = int(np.ceil(n_points ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:n_points]

def _optimize_from_initial(c_init, deadline):
    """Beta-continuation then high-precision polish from one initial config."""
    best_c = np.clip(c_init, -0.5, 0.5).copy()
    best_val = _compute_radius(best_c)

    betas = [5, 10, 20, 40, 80, 160, 320, 640, 1280]
    for beta in betas:
        if time.time() > deadline - 25:
            break
        try:
            res = minimize(_soft_min_objective_and_grad, best_c.ravel(),
                           args=(beta,), method='L-BFGS-B', jac=True,
                           bounds=[(-0.5, 0.5)] * (N * 3),
                           options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            new_c = np.clip(res.x.reshape(N, 3), -0.4995, 0.4995)
            new_val = _compute_radius(new_c)
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
        except Exception:
            pass

    # High-precision polish
    for beta_p in [5000, 10000]:
        if time.time() > deadline - 12:
            break
        try:
            res = minimize(_soft_min_objective_and_grad, best_c.ravel(),
                           args=(beta_p,), method='L-BFGS-B', jac=True,
                           bounds=[(-0.5, 0.5)] * (N * 3),
                           options={'maxiter': 600, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 30})
            new_c = np.clip(res.x.reshape(N, 3), -0.49999, 0.49999)
            new_val = _compute_radius(new_c)
            if new_val > best_val:
                best_val = new_val
                best_c = new_c.copy()
        except Exception:
            pass

    return best_c, best_val

def _exact_local_polish(c, deadline):
    """Subgradient ascent on the true nonsmooth radius for a final nudge."""
    best_c = c.copy()
    best_r = _compute_radius(best_c)
    lr = 0.02
    for _ in range(300):
        if time.time() > deadline - 3:
            break
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        mi, mj = np.unravel_index(np.argmin(dists), dists.shape)
        min_pair = dists[mi, mj] / 2.0
        bd = 0.5 - np.abs(c)
        bi, bk = np.unravel_index(np.argmin(bd), bd.shape)
        min_bound = bd[bi, bk]

        if min_pair < min_bound:
            i, j = mi, mj
            direction = c[i] - c[j]
            nrm = np.linalg.norm(direction)
            direction = direction / nrm if nrm > 1e-10 else np.array([1.0, 0, 0])
            grad = np.zeros_like(c)
            grad[i] += direction
            grad[j] -= direction
        else:
            grad = np.zeros_like(c)
            s = np.sign(c[bi, bk])
            grad[bi, bk] = -s if s != 0 else -1.0

        new_c = np.clip(c + lr * grad, -0.5, 0.5)
        new_r = _compute_radius(new_c)
        if new_r > best_r + 1e-12:
            c = new_c
            best_r = new_r
            best_c = new_c.copy()
            lr *= 0.95
        else:
            lr *= 0.8
            if lr < 1e-8:
                break
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before the 150s kill

    rng = np.random.default_rng(42)

    # Diverse initial configurations
    c_bcc = _generate_bcc_lattice(N)
    c_grid = _grid_initial(N)

    inits = [c_bcc.copy(), c_grid.copy()]
    for seed in [1, 2, 3]:
        inits.append(np.clip(c_bcc + rng.normal(0, 0.015, size=(N, 3)), -0.499, 0.499))
    inits.append(np.clip(c_grid + rng.normal(0, 0.015, size=(N, 3)), -0.499, 0.499))
    inits.append(rng.uniform(-0.4, 0.4, size=(N, 3)))

    best_radius = 0.0
    best_centers = None

    for c0 in inits:
        if time.time() > deadline - 30:
            break
        c, val = _optimize_from_initial(c0, deadline)
        if val > best_radius:
            best_radius = val
            best_centers = c.copy()

    # Final exact local polish on the global best
    if best_centers is not None and time.time() < deadline - 5:
        c, val = _exact_local_polish(best_centers, deadline)
        if val > best_radius:
            best_radius = val
            best_centers = c.copy()

    return best_centers
# EVOLVE-BLOCK-END
