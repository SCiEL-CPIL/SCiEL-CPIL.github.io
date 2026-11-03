"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

DEADLINE = 145.0  # safety margin before the 150s kill


def _true_score(c):
    """Exact packing radius in double precision."""
    c = np.asarray(c, dtype=float)
    if c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c < -0.5) or np.any(c > 0.5):
        return 0.0
    r = np.inf
    # wall distances
    w = np.min(np.stack([
        0.5 - c[:, 0], c[:, 0] + 0.5,
        0.5 - c[:, 1], c[:, 1] + 0.5,
    ]), axis=0)
    r = min(r, float(np.min(w)))
    # pairwise distances
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    r = min(r, float(np.min(d)) * 0.5)
    return r


def _violation_objective(x, r):
    """Sum of squared violations of minimum separation 2r and wall constraints.
    x is shape (N, 2). Returns (violation, gradient)."""
    c = x.reshape(N, 2)
    # Wall constraints: c must be in [-0.5+r, 0.5-r]
    lo = -0.5 + r
    hi = 0.5 - r
    
    # Compute violations
    viol = 0.0
    grad = np.zeros_like(c)
    
    # Wall violations
    for axis in range(2):
        # Below lower bound
        pen = lo - c[:, axis]
        mask = pen > 0
        if np.any(mask):
            viol += np.sum(pen[mask] ** 2)
            grad[mask, axis] += -2 * pen[mask]
        # Above upper bound
        pen = c[:, axis] - hi
        mask = pen > 0
        if np.any(mask):
            viol += np.sum(pen[mask] ** 2)
            grad[mask, axis] += 2 * pen[mask]
    
    # Pairwise violations: distance must be >= 2r
    diff = c[:, None, :] - c[None, :, :]
    dists = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dists, np.inf)
    
    mask = dists < 2 * r
    if np.any(mask):
        safe_dists = np.where(mask, np.maximum(dists, 1e-12), 1.0)
        overlap = np.where(mask, 2 * r - dists, 0.0)
        # Gradient contribution: for pair (i,j) with d_ij < 2r,
        # violation term is (2r - d_ij)^2, gradient w.r.t. c_i is -2(2r-d_ij)*(c_i-c_j)/d_ij
        vec = diff / safe_dists[:, :, None]
        grad_i = -2 * overlap[:, :, None] * vec
        grad += grad_i.sum(axis=1)
        viol += np.sum(overlap ** 2)
    
    return viol, grad


def _projected_gradient_descent(c0, r, time_budget=3.0, max_iter=50000, lr=0.01):
    """Projected gradient descent to minimize violations at fixed radius r.
    Returns feasible layout (all violations ~ 0) or best effort."""
    c = np.array(c0, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    
    best_c = c.copy()
    best_viol = _violation_objective(c, r)[0]
    
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        viol, grad = _violation_objective(c, r)
        
        if viol < 1e-12:
            break
        
        # Adaptive learning rate
        if it % 1000 == 0:
            lr = max(0.001, 0.01 * (0.5 ** (it // 5000)))
        
        c = c - lr * grad
        c = np.clip(c, -0.5 + r, 0.5 - r)
        
        new_viol = _violation_objective(c, r)[0]
        if new_viol < best_viol:
            best_viol = new_viol
            best_c = c.copy()
    
    return best_c


def _radius_bisection(c0, time_budget=10.0, r_lo=0.05, r_hi=0.08):
    """Bisect on radius r: find max r such that we can find a feasible packing.
    Uses projected gradient descent to find feasible layouts."""
    lo, hi = r_lo, r_hi
    best_c = c0.copy()
    best_r = _true_score(c0)
    
    start = time.time()
    
    # First, try to find a feasible layout at current best radius
    cur_r = best_r * 0.98
    for _ in range(5):
        if time.time() - start > time_budget - 2:
            break
        c = _projected_gradient_descent(c0, cur_r, time_budget=min(2.0, time_budget - 2 - (time.time() - start)))
        s = _true_score(c)
        if s >= cur_r - 1e-9:
            lo = cur_r
            if s > best_r:
                best_r, best_c = s, c
            cur_r = min(r_hi, cur_r * 1.03)
        else:
            hi = cur_r
            if hi - lo < 1e-6:
                break
            cur_r = 0.5 * (lo + hi)
    
    # Now do more careful bisection
    if time.time() - start < time_budget - 3:
        for _ in range(8):
            if time.time() - start > time_budget - 3:
                break
            mid = 0.5 * (lo + hi)
            c = _projected_gradient_descent(best_c, mid, time_budget=min(1.5, time_budget - 3 - (time.time() - start)))
            s = _true_score(c)
            if s >= mid - 1e-9:
                lo = mid
                if s > best_r:
                    best_r, best_c = s, c
            else:
                hi = mid
            if hi - lo < 1e-7:
                break
    
    return best_c, best_r


def _softmin_refine(c0, k, time_budget=3.0):
    """Smooth L-BFGS-B polish of a layout to raise the min radius."""
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)

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

    try:
        res = minimize(obj_and_grad, c0.ravel(), jac=True, method='L-BFGS-B',
                       bounds=[(-0.5, 0.5)] * (2 * N),
                       options={'maxiter': 300, 'maxfun': 8000,
                                'ftol': 1e-14, 'gtol': 1e-11,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _force_relax(c, r, time_budget=3.0, max_iter=10000):
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
        # pairwise overlap repulsion
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
        # wall push
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


def _hex_init():
    """Create a hexagonal-like initial arrangement for 50 circles."""
    pts = []
    # Use a 7x7 grid base (49 points) plus one extra
    m = 7
    for i in range(m):
        for j in range(m):
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / m - 0.5
            # Add small hexagonal offset
            if i % 2 == 1:
                x += 0.5 / m * 0.5
            pts.append((x, y))
    g = np.array(pts)
    # Add 50th point at a slightly offset position
    extra = np.array([-2.5 / 7.0, -2.5 / 7.0])
    return np.vstack([g, extra])


def _grid_init():
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in ticks for x in ticks])
    return np.vstack([g, np.array([-2.5 / 7.0, -2.5 / 7.0])])


def _random_init(rng):
    return rng.uniform(-0.42, 0.42, (N, 2))


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Multiple initialization strategies
    inits = [_hex_init(), _grid_init()]
    for _ in range(8):
        inits.append(_random_init(rng))
    
    best = inits[0].copy()
    best_r = _true_score(best)
    
    # Phase 1: For each init, do force relaxation + PGD radius bisection
    for idx, c0 in enumerate(inits):
        if time.time() - start_time > DEADLINE - 25:
            break
        
        # Quick force relaxation to get a decent starting point
        r_est = _true_score(c0) * 0.95
        r_est = max(r_est, 0.05)
        c_relaxed = _force_relax(c0, r_est, time_budget=1.5, max_iter=3000)
        s_relaxed = _true_score(c_relaxed)
        
        # PGD-based radius bisection
        c_pg, r_pg = _radius_bisection(c_relaxed, time_budget=8.0, r_lo=0.05, r_hi=0.075)
        
        if r_pg > best_r:
            best_r, best = r_pg, c_pg
        elif s_relaxed > best_r:
            best_r, best = s_relaxed, c_relaxed
    
    # Phase 2: Soft-min refinement on best and a few other candidates
    if time.time() - start_time < DEADLINE - 15:
        candidates = [best]
        # Add a few more candidates from inits
        for c0 in inits[:3]:
            c_rel = _force_relax(c0, best_r * 0.95, time_budget=1.0, max_iter=2000)
            candidates.append(c_rel)
        
        for k in (100.0, 200.0, 350.0):
            for c0 in candidates:
                if time.time() - start_time > DEADLINE - 10:
                    break
                c = _softmin_refine(c0, k, time_budget=2.5)
                s = _true_score(c)
                if s > best_r:
                    best_r, best = s, c
    
    # Phase 3: Final high-precision polish
    if time.time() - start_time < DEADLINE - 5:
        c = _softmin_refine(best, 500.0, time_budget=4.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c
    
    # Phase 4: One more PGD pass at the current best radius to try to improve
    if time.time() - start_time < DEADLINE - 3:
        r_try = best_r * 1.005
        c_pg, r_pg = _radius_bisection(best, time_budget=4.0, r_lo=best_r * 0.99, r_hi=r_try)
        if r_pg > best_r:
            best_r, best = r_pg, c_pg
    
    # Final soft-min polish
    if time.time() - start_time < DEADLINE - 2:
        c = _softmin_refine(best, 600.0, time_budget=3.0)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c
    
    return best
# EVOLVE-BLOCK-END
