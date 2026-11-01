"""
Packing 200 equal circles in the unit square [-1/2, 1/2]^2.

Combined strategy:
  1. Diverse initializations (hexagonal variants, grid, random).
  2. Global spread via L-BFGS-B on a smooth power-4 repulsive energy
     (sharp repulsion -> well-conditioned, pushes points to a near-uniform
     configuration so the true minimum distance is lifted).
  3. Gradient-based polish (moves every circle along the repulsion field).
  4. Bottleneck-targeted local search: identify the exact limiting
     constraint (the closest pair or the closest wall) and move it apart.
     This attacks the scored metric (min pair dist / 2 vs min wall dist)
     directly, which the surrogate energy only approximates.
  5. Pair-repulse: push the whole "close pair cluster" apart to relieve
     multiple near-equal constraints simultaneously.
  6. Final high-precision bottleneck polish.

Score = min( min_{i<j} |c_i-c_j|/2 , min_i min_k (1/2 - |c_ik|) ).
"""

N = 200
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def _compute_radius(c):
    """Largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0

    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))

    if n < 2:
        return min_boundary

    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists) if len(dists) > 0 else np.inf

    return min(min_pair_dist / 2.0, min_boundary)


def _energy_and_grad(params, power=4.0):
    """
    Smooth repulsive energy + analytic gradient.
    E_pair = sum_{i<j} 1/d_ij^4 ; E_wall = sum_i sum_wall 1/d_iw^2
    """
    c = params.reshape(N, 2)
    x = c[:, 0]
    y = c[:, 1]

    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    d2 = dx**2 + dy**2
    np.fill_diagonal(d2, 1.0)

    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    d2_neg2 = 1.0 / (d2_safe**2)
    d2_neg3 = 1.0 / (d2_safe**3)
    np.fill_diagonal(d2_neg2, 0.0)
    np.fill_diagonal(d2_neg3, 0.0)

    E_pair = np.triu(d2_neg2, 1).sum()

    grad_x = -4.0 * (dx * d2_neg3).sum(axis=1)
    grad_y = -4.0 * (dy * d2_neg3).sum(axis=1)

    eps = 1e-8
    dx_w = np.clip(0.5 - x, eps, None)
    dx_w_neg = np.clip(0.5 + x, eps, None)
    dy_w = np.clip(0.5 - y, eps, None)
    dy_w_neg = np.clip(0.5 + y, eps, None)

    E_wall = (1.0/dx_w**2 + 1.0/dx_w_neg**2 + 1.0/dy_w**2 + 1.0/dy_w_neg**2).sum()

    grad_x += 2.0 / dx_w**3 - 2.0 / dx_w_neg**3
    grad_y += 2.0 / dy_w**3 - 2.0 / dy_w_neg**3

    total_E = E_pair + E_wall
    total_grad = np.concatenate([grad_x, grad_y])

    return total_E, total_grad


def _initialize_hexagonal(d_spacing=None):
    """Hexagonal pattern with given spacing (returns None if < N points)."""
    if d_spacing is None:
        d_spacing = 0.07

    points = []
    row_spacing = d_spacing * np.sqrt(3) / 2
    x_offset = d_spacing / 2

    y = -0.5 + d_spacing / 2
    row = 0
    while y <= 0.5 - d_spacing / 2:
        x = -0.5 + d_spacing / 2
        if row % 2 == 1:
            x += x_offset
        while x <= 0.5 - d_spacing / 2:
            points.append([x, y])
            x += d_spacing
        y += row_spacing
        row += 1

    points = np.array(points)
    if len(points) >= N:
        c = points[:N].copy()
        c = np.clip(c, -0.49, 0.49)
        return c
    return None


def _initialize_grid():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initialize_random(seed=42):
    rng = np.random.RandomState(seed)
    c = rng.uniform(-0.45, 0.45, size=(N, 2))
    return c


def _bottleneck(c):
    """
    Return (r, kind, idx) where kind is 'pair' (idx=(i,j)) or 'wall'
    (idx=(circle, axis, sign)) and r is the limiting radius value.
    Uses pdist for the pair term (fast and exact).
    """
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)

    dists_x = 0.5 - np.abs(c[:, 0])
    dists_y = 0.5 - np.abs(c[:, 1])
    min_boundary = np.minimum(np.min(dists_x), np.min(dists_y))

    if min_pair_dist / 2.0 < min_boundary:
        # pair is limiting -> find (i, j)
        min_idx = int(np.argmin(dists))
        # condensed index -> (i, j), i < j
        # k = i*N - i*(i+1)//2 + (j - i - 1)
        i = int(np.floor((-1 + np.sqrt(1 + 8 * min_idx)) / 2))
        j = i + 1 + (min_idx - (i * N - i * (i + 1) // 2))
        return min_pair_dist / 2.0, 'pair', (i, j)
    else:
        # wall is limiting -> find circle and nearest wall
        bdist = np.minimum(np.minimum(0.5 - c[:, 0], 0.5 + c[:, 0]),
                           np.minimum(0.5 - c[:, 1], 0.5 + c[:, 1]))
        ci = int(np.argmin(bdist))
        # determine which wall
        d_left = 0.5 - c[ci, 0]
        d_right = 0.5 + c[ci, 0]
        d_bottom = 0.5 - c[ci, 1]
        d_top = 0.5 + c[ci, 1]
        cand = [(d_left, 0, -1), (d_right, 0, +1), (d_bottom, 1, -1), (d_top, 1, +1)]
        d, axis, sign = min(cand, key=lambda t: t[0])
        return min_boundary, 'wall', (ci, axis, sign)


def _local_polish(c, max_iters=5000, time_limit=25.0, base_step=1e-4):
    """Gradient-field polish: move every circle along the repulsion gradient."""
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    step = base_step
    stall_count = 0
    max_stall = 50

    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        _, grad = _energy_and_grad(c.flatten(), power=4.0)
        grad = grad.reshape(N, 2)

        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.where(grad_norm > 1e-12, grad_norm, 1.0)
        grad_dir = grad / grad_norm

        new_c = np.clip(c + step * grad_dir, -0.499, 0.499)
        new_r = _compute_radius(new_c)
        if new_r > current_r + 1e-12:
            c = new_c
            current_r = new_r
            stall_count = 0
            step = min(step * 1.05, 5e-4)
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        else:
            improved = False
            for factor in [0.5, 0.25, 0.1]:
                new_c2 = np.clip(c + step * factor * grad_dir, -0.499, 0.499)
                new_r2 = _compute_radius(new_c2)
                if new_r2 > current_r + 1e-12:
                    c = new_c2
                    current_r = new_r2
                    stall_count = 0
                    step = min(step * 1.02, 5e-4)
                    if current_r > best_r:
                        best_r = current_r
                        best_c = c.copy()
                    improved = True
                    break
            if not improved:
                stall_count += 1
                step = max(step * 0.8, 1e-6)
                if stall_count > max_stall:
                    break

    return best_c, best_r


def _bottleneck_polish(c, max_iters=20000, time_limit=20.0, base_step=1e-4,
                       cluster_frac=0.985):
    """
    Exact bottleneck-targeted local search.
    Directly maximizes the scored metric by moving the limiting constraint
    apart, and by pushing the whole close-pair cluster apart.
    """
    start_time = time.time()
    c = c.copy()
    current_r = _compute_radius(c)
    best_c = c.copy()
    best_r = current_r
    step = base_step
    stall = 0
    max_stall = 60

    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break

        r, kind, idx = _bottleneck(c)

        new_c = c
        improved = False

        if kind == 'pair':
            i, j = idx
            vec = c[i] - c[j]
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                u = vec / norm
                # (a) move the limiting pair apart
                cand = c.copy()
                cand[i] += u * step
                cand[j] -= u * step
                cand = np.clip(cand, -0.499, 0.499)
                cr = _compute_radius(cand)
                if cr > current_r + 1e-13:
                    new_c = cand
                    improved = True
                else:
                    # (b) push the whole close-pair cluster apart
                    dists = pdist(c, metric='euclidean')
                    thresh = max(cluster_frac * 2.0 * current_r, 1e-9)
                    # build displacement: for every close pair, push both ends apart
                    disp = np.zeros_like(c)
                    # iterate close pairs via condensed index
                    # (N=200 -> 19900 pairs; vectorized enough for a limited stage)
                    ii, jj = np.triu_indices(N, 1)
                    close = dists < thresh
                    if np.any(close):
                        dvec = c[ii[close]] - c[jj[close]]
                        dn = np.linalg.norm(dvec, axis=1)
                        dn = np.where(dn > 1e-12, dn, 1.0)
                        unit = dvec / dn[:, None]
                        # accumulate
                        np.add.at(disp, ii[close], unit * step)
                        np.add.at(disp, jj[close], -unit * step)
                        cand2 = np.clip(c + disp, -0.499, 0.499)
                        cr2 = _compute_radius(cand2)
                        if cr2 > current_r + 1e-13:
                            new_c = cand2
                            improved = True

        else:
            ci, axis, sign = idx
            # move circle ci away from the limiting wall (toward centre)
            dvec = np.zeros(2)
            dvec[axis] = -sign * step
            cand = c.copy()
            cand[ci] += dvec
            cand = np.clip(cand, -0.499, 0.499)
            cr = _compute_radius(cand)
            if cr > current_r + 1e-13:
                new_c = cand
                improved = True

        if improved:
            c = new_c
            current_r = _compute_radius(c)
            stall = 0
            step = min(step * 1.03, 2e-4)
            if current_r > best_r:
                best_r = current_r
                best_c = c.copy()
        else:
            stall += 1
            step = max(step * 0.85, 1e-7)
            if stall > max_stall:
                break

    return best_c, best_r


def _lbfgs(c, maxiter, maxfun, deadline):
    bounds = [(-0.499, 0.499)] * (2 * N)
    res = minimize(
        _energy_and_grad,
        c.flatten(),
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12,
                 'maxfun': maxfun, 'maxls': 50}
    )
    c_opt = np.clip(res.x.reshape(N, 2), -0.499, 0.499)
    return c_opt, _compute_radius(c_opt)


def construct():
    start_time = time.time()
    deadline = start_time + 145  # hard kill is 150s; keep a 5s safety margin

    # ---- 1. Diverse initializations -------------------------------------
    inits = []
    for d in np.linspace(0.065, 0.075, 5):
        c_hex = _initialize_hexagonal(d)
        if c_hex is not None:
            inits.append(c_hex)
    inits.append(_initialize_grid())
    for seed in [42, 123, 456]:
        inits.append(_initialize_random(seed))

    init_results = []
    for c_init in inits:
        c_init = np.clip(c_init, -0.49, 0.49)
        init_results.append((_compute_radius(c_init), c_init))
    init_results.sort(key=lambda x: x[0], reverse=True)
    top_inits = [c for _, c in init_results[:3]]

    best_c = top_inits[0].copy()
    best_r = _compute_radius(best_c)

    # ---- 2. Global spread via L-BFGS-B (best init) ----------------------
    if time.time() < deadline - 55:
        try:
            c_opt, r_opt = _lbfgs(best_c, 500, 5000, deadline)
            if r_opt > best_r:
                best_r, best_c = r_opt, c_opt.copy()
        except Exception:
            pass

    # ---- 3. Second-best init, L-BFGS-B ----------------------------------
    if time.time() < deadline - 40 and len(top_inits) > 1:
        try:
            c_opt2, r_opt2 = _lbfgs(top_inits[1], 300, 3000, deadline)
            if r_opt2 > best_r:
                best_r, best_c = r_opt2, c_opt2.copy()
        except Exception:
            pass

    # ---- 4. Gradient-field polish ---------------------------------------
    if time.time() < deadline - 30:
        t_polish = min(20.0, deadline - time.time() - 30)
        if t_polish > 2.0:
            c_p, r_p = _local_polish(best_c, max_iters=10000,
                                     time_limit=t_polish, base_step=2e-4)
            if r_p > best_r:
                best_r, best_c = r_p, c_p.copy()

    # ---- 5. Bottleneck-targeted exact polish ----------------------------
    if time.time() < deadline - 15:
        t_bn = min(20.0, deadline - time.time() - 15)
        if t_bn > 2.0:
            c_b, r_b = _bottleneck_polish(best_c, max_iters=20000,
                                          time_limit=t_bn, base_step=1e-4)
            if r_b > best_r:
                best_r, best_c = r_b, c_b.copy()

    # ---- 6. Final high-precision bottleneck polish ----------------------
    if time.time() < deadline - 4:
        t_final = min(8.0, deadline - time.time() - 4)
        if t_final > 1.0:
            c_f, r_f = _bottleneck_polish(best_c, max_iters=15000,
                                          time_limit=t_final, base_step=4e-5)
            if r_f > best_r:
                best_r, best_c = r_f, c_f.copy()

    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
