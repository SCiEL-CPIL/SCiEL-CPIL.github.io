"""Combined packing: Voronoi/Delaunay-based force relaxation + SLSQP epigraph + soft-min polish."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree, Delaunay

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


def _voronoi_inradius_relax(c, r, time_budget=4.0, max_iter=5000):
    """
    Structural approach: use Voronoi/Delaunay structure to identify which
    constraints are active (binding) and apply targeted forces.
    
    For each circle i, its Voronoi cell is bounded by:
    - Midlines to Delaunay neighbors (pairwise constraints)
    - Square walls (wall constraints)
    
    The inradius of the Voronoi cell is the minimum distance from center i
    to any cell boundary. We want this inradius >= r for all i.
    
    This gives us a geometrically-aware force model: instead of generic
    repulsion, we push each center away from its binding Voronoi boundary.
    """
    c = np.array(c, dtype=float, copy=True)
    c = np.clip(c, -0.5 + r, 0.5 - r)
    start = time.time()
    dt = 0.03
    
    for _ in range(max_iter):
        if time.time() - start > time_budget:
            break
        
        disp = np.zeros_like(c)
        
        # Build Delaunay triangulation to find neighbors
        try:
            tri = Delaunay(c)
            neighbors = [set() for _ in range(N)]
            for simplex in tri.simplices:
                for idx in simplex:
                    for other in simplex:
                        if idx != other:
                            neighbors[idx].add(other)
        except Exception:
            neighbors = [set(range(N)) - {i} for i in range(N)]
        
        # For each circle, compute forces from its Voronoi cell boundaries
        for i in range(N):
            # Wall forces
            for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1)]:
                if sign > 0:
                    pen = c[i, axis] - (0.5 - r)
                    pen = max(pen, 0)
                else:
                    pen = (-0.5 + r) - c[i, axis]
                    pen = max(pen, 0)
                if pen > 0:
                    disp[i, axis] += sign * pen / (2 * r)
            
            # Neighbor forces (Voronoi midline-based)
            for j in neighbors[i]:
                if j <= i:
                    continue
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    continue
                if d < 2 * r:
                    overlap = (2 * r - d) / (2 * r)
                    force = overlap * diff / d
                    disp[i] += force
                    disp[j] -= force
        
        c = np.clip(c + dt * disp, -0.5 + r, 0.5 - r)
    
    return c


def _softmin_refine(c0, k, time_budget=3.0, maxiter=200, maxfun=5000,
                    ftol=1e-15, gtol=1e-10):
    """Smooth L-BFGS-B polish of a feasible layout to raise the min radius."""
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
                       options={'maxiter': maxiter, 'maxfun': maxfun,
                                'ftol': ftol, 'gtol': gtol,
                                'iprint': 0})
        c = np.clip(res.x.reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def _slsqp_epigraph(c0, r_init, time_budget=20.0, max_iter=300, ftol=1e-12):
    """
    SLSQP with linear epigraph formulation:
    maximize t subject to:
      t <= |c_i - c_j| / 2  for all i < j
      t <= 0.5 - c[i,k]     for all i, k
      t <= c[i,k] + 0.5     for all i, k
    
    This directly optimizes the packing radius with explicit constraints.
    """
    c0 = np.clip(np.asarray(c0, dtype=float), -0.5, 0.5)
    r_init = max(r_init, 0.01)
    
    # Initial guess: [centers..., t]
    x0 = np.concatenate([c0.ravel(), [r_init]])
    
    def neg_t(x):
        return -x[2 * N]

    def neg_t_grad(x):
        g = np.zeros(2 * N + 1)
        g[2 * N] = -1.0
        return g

    cons = []
    # Pairwise constraints: |c_i - c_j|/2 - t >= 0
    for i in range(N):
        for j in range(i + 1, N):
            def pair_con(x, i=i, j=j):
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                t = x[2 * N]
                d = np.sqrt(np.sum((ci - cj) ** 2))
                return d / 2.0 - t
            def pair_con_grad(x, i=i, j=j):
                g = np.zeros(2 * N + 1)
                ci = x[2 * i:2 * i + 2]
                cj = x[2 * j:2 * j + 2]
                t = x[2 * N]
                diff = ci - cj
                d = np.sqrt(np.sum(diff ** 2))
                if d > 1e-12:
                    g[2 * i:2 * i + 2] = diff / (2 * d)
                    g[2 * j:2 * j + 2] = -diff / (2 * d)
                g[2 * N] = -1.0
                return g
            cons.append({'type': 'ineq', 'fun': pair_con, 'jac': pair_con_grad})
    
    # Wall constraints
    for i in range(N):
        # Upper x: 0.5 - c[i,0] - t >= 0
        def wall_ux(x, i=i):
            return 0.5 - x[2 * i] - x[2 * N]
        def wall_ux_grad(x, i=i):
            g = np.zeros(2 * N + 1)
            g[2 * i] = -1.0
            g[2 * N] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': wall_ux, 'jac': wall_ux_grad})
        
        # Lower x: c[i,0] + 0.5 - t >= 0
        def wall_lx(x, i=i):
            return x[2 * i] + 0.5 - x[2 * N]
        def wall_lx_grad(x, i=i):
            g = np.zeros(2 * N + 1)
            g[2 * i] = 1.0
            g[2 * N] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': wall_lx, 'jac': wall_lx_grad})
        
        # Upper y: 0.5 - c[i,1] - t >= 0
        def wall_uy(x, i=i):
            return 0.5 - x[2 * i + 1] - x[2 * N]
        def wall_uy_grad(x, i=i):
            g = np.zeros(2 * N + 1)
            g[2 * i + 1] = -1.0
            g[2 * N] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': wall_uy, 'jac': wall_uy_grad})
        
        # Lower y: c[i,1] + 0.5 - t >= 0
        def wall_ly(x, i=i):
            return x[2 * i + 1] + 0.5 - x[2 * N]
        def wall_ly_grad(x, i=i):
            g = np.zeros(2 * N + 1)
            g[2 * i + 1] = 1.0
            g[2 * N] = -1.0
            return g
        cons.append({'type': 'ineq', 'fun': wall_ly, 'jac': wall_ly_grad})
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    start = time.time()
    try:
        res = minimize(neg_t, x0, jac=neg_t_grad, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': max_iter, 'ftol': ftol,
                                'iprint': 0})
        if time.time() - start > time_budget:
            return c0
        c = np.clip(res.x[:2 * N].reshape(N, 2), -0.5, 0.5)
    except Exception:
        c = c0
    return c


def construct():
    start_time = time.time()
    best = _grid_init()
    best_r = _true_score(best)

    inits = [_grid_init(), _hex_init()]
    rng = np.random.default_rng(12345)
    for _ in range(6):
        inits.append(rng.uniform(-0.42, 0.42, (N, 2)))

    # 1) Hard-disk radius expansion from several starts using both
    #    generic force relaxation and Voronoi-aware relaxation.
    for c0 in inits:
        if time.time() - start_time > DEADLINE - 15:
            break
        r = _true_score(c0) * 0.95
        r = max(r, 0.05)
        lo, hi = 0.0, r
        cur = c0
        for _ in range(8):
            if time.time() - start_time > DEADLINE - 15:
                break
            # Alternate between generic and Voronoi-aware relaxation
            if _ % 2 == 0:
                cur = _force_relax(cur, r, time_budget=1.5, max_iter=3000)
            else:
                cur = _voronoi_inradius_relax(cur, r, time_budget=1.5, max_iter=2000)
            s = _true_score(cur)
            if s >= r - 1e-9:
                lo = r
                r = r * 1.05
                hi = r
            else:
                hi = r
                if hi - lo < 1e-6:
                    break
                r = 0.5 * (lo + hi)
        s = _true_score(cur)
        if s > best_r:
            best_r, best = s, cur

    # 2) SLSQP epigraph optimization from best layout and a few others
    candidates = [best]
    candidates.append(_force_relax(best, best_r, time_budget=1.0, max_iter=1500))
    for c0 in inits[:3]:
        candidates.append(_force_relax(c0, best_r, time_budget=1.0, max_iter=1500))

    for c0 in candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
        c = _slsqp_epigraph(c0, best_r, time_budget=15.0, max_iter=200, ftol=1e-12)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 3) Smooth soft-min polish
    for k in (80.0, 150.0, 250.0, 400.0):
        if time.time() - start_time > DEADLINE - 6:
            break
        c = _softmin_refine(best, k, time_budget=2.5, maxiter=150, maxfun=4000)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 4) Final high-precision polish
    if time.time() - start_time < DEADLINE - 3:
        c = _softmin_refine(best, 300.0, time_budget=3.0,
                            maxiter=300, maxfun=8000,
                            ftol=1e-16, gtol=1e-12)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    # 5) Ultra-high-precision final pass
    if time.time() - start_time < DEADLINE - 1:
        c = _softmin_refine(best, 500.0, time_budget=2.0,
                            maxiter=200, maxfun=6000,
                            ftol=1e-18, gtol=1e-13)
        s = _true_score(c)
        if s > best_r:
            best_r, best = s, c

    return best
# EVOLVE-BLOCK-END
