"""Combined packing program: SLSQP linear-epigraph primary + multi-stage refinement."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DEADLINE = 118.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = np.inf
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _hex_init():
    m = 7
    pts = []
    for i in range(m):
        for j in range(m):
            pts.append(((j + 0.5) / m - 0.5, (i + 0.5) / m - 0.5))
    g = np.array(pts)
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _force_relax(c, r, time_budget=3.0, max_iter=15000):
    """Grow/relax a hard-disk packing: push overlapping circles apart and
    keep them inside the square. Returns a feasible layout for radius r."""
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.05
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        disp = np.zeros_like(c)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        mask = d < 2 * r
        if np.any(mask):
            safe = np.where(d > 1e-12, d, 1e-12)
            vec = (c[:, None, :] - c[None, :, :]) / safe[:, :, None]
            overlap = (2 * r - d) / (2 * r)
            overlap = np.where(mask, overlap, 0.0)
            force = overlap[:, :, None] * vec
            disp += force.sum(axis=1)
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    return c


def _langevin_dynamics(c0, r, time_budget=6.0, max_iter=100000, rng=None):
    """Langevin dynamics for hard-disk packing at fixed radius r."""
    if rng is None:
        rng = np.random.default_rng(42)
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    v = np.zeros_like(c)
    start = time.time()
    dt = 0.002
    gamma = 0.1
    T_init = 0.01
    T_min = 0.00001
    n = max_iter
    best_c = c.copy()
    best_score = _true_score(c)
    for i in range(max_iter):
        if time.time() - start > time_budget:
            break
        T = T_init * (T_min / T_init) ** (i / max(1, n))
        disp = np.zeros_like(c)
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * r + 1e-6, output_type='ndarray')
        if len(pairs) > 0:
            for idx in pairs:
                i, j = int(idx[0]), int(idx[1])
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    diff = np.array([1e-8, 0.0])
                    d = 1e-8
                if d < 2 * r:
                    overlap = (2 * r - d) / (2 * r)
                    force = overlap * diff / d
                    disp[i] += force
                    disp[j] -= force
        for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
            if sign > 0:
                pen = c[:, axis] - (0.5 - r)
                pen = np.clip(pen, 0, None)
            else:
                pen = (-0.5 + r) - c[:, axis]
                pen = np.clip(pen, 0, None)
            disp[:, axis] += sign * pen / (2 * r)
        noise = rng.normal(0, 1, size=c.shape)
        v = (v - dt * disp) * np.exp(-gamma * dt) + np.sqrt(2 * T * dt * gamma) * noise
        c_new = c + dt * v
        c_new = np.clip(c_new, -0.5 + r, 0.5 - r)
        tree_new = cKDTree(c_new)
        pairs_new = tree_new.query_pairs(2 * r - 1e-9, output_type='ndarray')
        feasible = len(pairs_new) == 0
        if feasible:
            score = _true_score(c_new)
            if score >= r - 1e-9:
                c = c_new
                if score > best_score:
                    best_score = score
                    best_c = c.copy()
            else:
                if np.random.random() < np.exp((score - r) / max(T, 1e-12)):
                    c = c_new
        else:
            v *= 0.5
    return best_c


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish with adaptive k schedule."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    start_time = time.time()

    def obj_and_grad(x):
        pts = x.reshape(-1, 2)
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dists, np.inf)
        dw = np.stack([
            0.5 - pts[:, 0], 0.5 + pts[:, 0],
            0.5 - pts[:, 1], 0.5 + pts[:, 1],
        ])
        all_d = np.concatenate([dists.ravel(), dw.ravel()])
        all_d = np.clip(all_d, 1e-8, 1.0)
        weights = np.exp(-k * all_d)
        sw = weights.sum()
        val = -np.log(sw) / k
        grad_pts = np.zeros_like(pts)
        wp = weights[:N * N].reshape(N, N)
        mask = ~np.eye(N, dtype=bool)
        safe = np.where(mask, dists, 1.0)
        factors = np.where(mask, wp / sw / safe, 0.0)
        s = factors.sum(axis=1)
        wpts = (factors[..., None] * pts[None, :, :]).sum(axis=1)
        grad_pts += pts * s[:, None] - wpts
        ww = weights[N * N:].reshape(4, N) / sw
        grad_pts[:, 0] += -ww[0].sum() + ww[1].sum()
        grad_pts[:, 1] += -ww[2].sum() + ww[3].sum()
        return val, -grad_pts.ravel()

    best_c = c0.copy()
    best_r = _true_score(c0)

    # Adaptive k: start broad, narrow down
    for k_val in [k * 0.5, k, k * 2.0]:
        if time.time() - start_time > time_budget * 0.8:
            break
        try:
            res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                           bounds=[(-0.5, 0.5)] * (2 * N),
                           options={'maxiter': 100, 'maxfun': 2000,
                                    'ftol': 1e-14, 'gtol': 1e-10,
                                    'iprint': 0})
            c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
            s = _true_score(c)
            if s > best_r:
                best_r = s
                best_c = c.copy()
                c0 = c  # Continue from improved point
        except Exception:
            pass

    return best_c


def _slsqp_linear_epigraph(c0, r_init, time_budget=15.0, max_iter=400, ftol=1e-13):
    """SLSQP with vectorized constraints for speed."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.01)

    # Precompute pair indices as arrays for vectorized ops
    i_idx, j_idx = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    pair_i = i_idx[mask]
    pair_j = j_idx[mask]
    n_pairs = len(pair_i)

    def neg_t(x):
        return -x[2 * N]

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    def pair_constraints(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[2 * N]
        diff = pts[pair_i] - pts[pair_j]
        d = np.sqrt(np.sum(diff ** 2, axis=1))
        return d / 2.0 - t

    def pair_constraints_grad(x):
        pts = x[:2 * N].reshape(N, 2)
        diff = pts[pair_i] - pts[pair_j]
        d = np.sqrt(np.sum(diff ** 2, axis=1))
        d_safe = np.where(d > 1e-12, d, 1.0)
        # grad w.r.t. pts[i] = diff / (2*d), grad w.r.t. pts[j] = -diff / (2*d)
        # Build sparse-like: accumulate into full grad array
        grad = np.zeros((n_pairs, 2 * N + 1))
        # For each pair k: grad[k, 2*pair_i[k]:2*pair_i[k]+2] = diff[k]/(2*d[k])
        # Use advanced indexing
        grad[:, 2 * N] = -1.0
        # Accumulate contributions
        # We need to scatter: for each pair k, add diff[k]/(2*d[k]) to row k, cols 2*pair_i[k], 2*pair_i[k]+1
        # and -diff[k]/(2*d[k]) to cols 2*pair_j[k], 2*pair_j[k]+1
        scale = 1.0 / (2.0 * d_safe)
        # Build index arrays
        ii = pair_i * 2
        jj = pair_j * 2
        # Scatter using np.add.at
        np.add.at(grad, (np.arange(n_pairs), ii), diff[:, 0] * scale)
        np.add.at(grad, (np.arange(n_pairs), ii + 1), diff[:, 1] * scale)
        np.add.at(grad, (np.arange(n_pairs), jj), -diff[:, 0] * scale)
        np.add.at(grad, (np.arange(n_pairs), jj + 1), -diff[:, 1] * scale)
        return grad

    def wall_constraints(x):
        pts = x[:2 * N].reshape(N, 2)
        t = x[2 * N]
        upper = 0.5 - pts - t
        lower = pts + 0.5 - t
        return np.concatenate([upper.ravel(), lower.ravel()])

    def wall_constraints_grad(x):
        # Precompute once (constant)
        grad = np.zeros((4 * N, 2 * N + 1))
        for i in range(N):
            grad[i, 2 * i] = -1.0
            grad[i, 2 * i + 1] = -1.0
            grad[i, 2 * N] = -1.0
            grad[N + i, 2 * i] = 1.0
            grad[N + i, 2 * i + 1] = 1.0
            grad[N + i, 2 * N] = -1.0
        return grad

    # Warm-start t slightly above current best to encourage growth
    t0 = min(r_init * 1.005, 0.5)
    x0 = np.concatenate([c0.ravel(), [t0]])
    start = time.time()
    best_c = c0.copy()
    best_r = _true_score(c0)

    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=[(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)],
                       constraints=[
                           {'type': 'ineq', 'fun': pair_constraints, 'jac': pair_constraints_grad},
                           {'type': 'ineq', 'fun': wall_constraints, 'jac': wall_constraints_grad},
                       ],
                       options={'maxiter': max_iter, 'ftol': ftol, 'disp': False})
        if res.x is not None and np.all(np.isfinite(res.x)):
            c_res = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
            s = _true_score(c_res)
            if s > best_r:
                best_r, best_c = s, c_res
    except Exception:
        pass

    return best_c


def _contact_graph_refine(c0, r_target, time_budget=3.0, max_iter=300):
    """Contact-graph-aware local optimization."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r_target, 0.5 - r_target)
    start = time.time()
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        wall_dists = np.stack([
            0.5 - c[:, 0], c[:, 0] + 0.5,
            0.5 - c[:, 1], c[:, 1] + 0.5,
        ], axis=0)
        min_pair = d.min()
        min_wall = wall_dists.min()
        tol = 1e-4
        active_pairs = np.argwhere(d < min_pair + tol)
        active_pairs = [(int(i), int(j)) for i, j in active_pairs if i < j]
        active_walls = []
        for k in range(N):
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                if sign > 0:
                    dist = 0.5 - c[k, axis]
                else:
                    dist = c[k, axis] + 0.5
                if dist < min_wall + tol:
                    active_walls.append((k, axis, sign))
        if len(active_pairs) == 0 and len(active_walls) == 0:
            break
        n_vars = 2 * N
        A = []
        for i, j in active_pairs:
            diff = c[i] - c[j]
            row = np.zeros(n_vars)
            row[2 * i:2 * i + 2] = diff
            row[2 * j:2 * j + 2] = -diff
            A.append(row)
        for k, axis, sign in active_walls:
            row = np.zeros(n_vars)
            row[2 * k + axis] = sign
            A.append(row)
        if len(A) == 0:
            break
        A = np.array(A)
        disp = np.zeros(n_vars)
        for row in A:
            disp += row
        norm = np.linalg.norm(disp)
        if norm < 1e-12:
            break
        disp = disp / norm
        best_c = c.copy()
        best_r = _true_score(c)
        for step in [0.001, 0.005, 0.01, 0.02, 0.05]:
            c_try = c + step * disp.reshape(N, 2)
            c_try = np.clip(c_try, -0.5 + r_target * 0.9, 0.5 - r_target * 0.9)
            s = _true_score(c_try)
            if s > best_r:
                best_r = s
                best_c = c_try.copy()
        c = best_c
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(8):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 40:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(10):
            if time.time() - start_time > DEADLINE - 40:
                break
            cur = _force_relax(cur, r, time_budget=1.0, max_iter=2000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.06
                hi = r
            else:
                hi = r
                if hi - lo < 1e-7:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) Langevin dynamics to escape local minima at current best radius
    if time.time() - start_time < DEADLINE - 35:
        ld_budget = min(8.0, DEADLINE - 35 - (time.time() - start_time))
        if ld_budget > 3.0:
            rng_ld = np.random.default_rng(777)
            cur = _langevin_dynamics(best, best_r, time_budget=ld_budget, max_iter=80000, rng=rng_ld)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 3) Primary SLSQP with LINEAR epigraph: the main optimizer
    if time.time() - start_time < DEADLINE - 20:
        slsqp_budget = min(30.0, DEADLINE - 20 - (time.time() - start_time))
        if slsqp_budget > 5.0:
            cur = _slsqp_linear_epigraph(best, best_r, time_budget=slsqp_budget, max_iter=400, ftol=1e-13)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 4) Soft-min polish from the best layout
    if time.time() - start_time < DEADLINE - 15:
        for k in (80.0, 150.0, 300.0):
            if time.time() - start_time > DEADLINE - 12:
                break
            c = _softmin_refine(best, k, time_budget=2.5)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 5) Contact-graph-aware refinement
    if time.time() - start_time < DEADLINE - 10:
        cg_budget = min(3.0, DEADLINE - 10 - (time.time() - start_time))
        if cg_budget > 1.0:
            cur = _contact_graph_refine(best, best_r, time_budget=cg_budget)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 6) Second SLSQP pass with tighter tolerance for final precision
    if time.time() - start_time < DEADLINE - 10:
        slsqp2_budget = min(25.0, DEADLINE - 10 - (time.time() - start_time))
        if slsqp2_budget > 5.0:
            cur = _slsqp_linear_epigraph(best, best_r, time_budget=slsqp2_budget, max_iter=600, ftol=1e-15)
            s = _true_score(cur)
            if s > best_r:
                best_r, best = s, cur

    # 7) Final high-precision soft-min polish with very large k
    if time.time() - start_time < DEADLINE - 4:
        for k_val in (400.0, 800.0):
            if time.time() - start_time > DEADLINE - 3:
                break
            c = _softmin_refine(best, k=k_val, time_budget=2.0)
            s = _true_score(c)
            if s > best_r:
                best_r, best = s, c

    # 8) Final contact-graph pass
    if time.time() - start_time < DEADLINE - 1:
        cur = _contact_graph_refine(best, best_r, time_budget=1.5)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 9) Ultra-precise final SLSQP polish on active constraints
    if time.time() - start_time < DEADLINE - 0.5:
        cur = _slsqp_linear_epigraph(best, best_r, time_budget=1.0, max_iter=100, ftol=1e-16)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    return best
# EVOLVE-BLOCK-END
