"""Packing of 50 equal circles in a unit square via soft-min epigraph + multi-start + L-BFGS polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5


def _compute_r(pts):
    """Compute the feasible radius for a configuration (double precision)."""
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _soft_min_epigraph_obj(x, beta):
    """
    Epigraph formulation: minimize -t + sum of soft violations.
    x = [pts_flat (2N), t]
    """
    pts = x[:2 * N].reshape(N, 2)
    t = x[2 * N]
    
    # Pairwise soft violations: max(0, 2t - dist_ij)
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_viol = np.maximum(0.0, 2.0 * t - pair_dist)
    # Soft sum: log-sum-exp for smoothness
    if np.any(pair_viol > 0):
        max_v = np.max(pair_viol)
        lse = max_v + np.log(np.sum(np.exp(pair_viol - max_v)))
        soft_pair = lse / beta
    else:
        soft_pair = 0.0
    
    # Wall soft violations: max(0, t - d_wall)
    d_wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                        np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, t - d_wall)
    if np.any(wall_viol > 0):
        max_v = np.max(wall_viol)
        lse_w = max_v + np.log(np.sum(np.exp(wall_viol - max_v)))
        soft_wall = lse_w / beta
    else:
        soft_wall = 0.0
    
    return -t + soft_pair + soft_wall


def _soft_min_epigraph_grad(x, beta):
    """Gradient of the epigraph objective."""
    pts = x[:2 * N].reshape(N, 2)
    t = x[2 * N]
    
    grad_pts = np.zeros_like(pts)
    grad_t = -1.0
    
    # Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)
    dist_safe = np.where(dist > 1e-12, dist, 1e-12)
    iu = np.triu_indices(N, k=1)
    pair_dist = dist[iu]
    pair_viol = np.maximum(0.0, 2.0 * t - pair_dist)
    
    if np.any(pair_viol > 0):
        max_v = np.max(pair_viol)
        exp_v = np.exp(pair_viol - max_v)
        weights = exp_v / np.sum(exp_v) / beta
        
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = weights[idx]
            d = dist_safe[i, j]
            diff_ij = (pts[i] - pts[j]) / d
            # d(pair_viol)/d(pts_i) = -diff_ij/d * (2t - dist) > 0 means gradient of viol wrt pts_i
            # Actually: viol = max(0, 2t - dist), d(viol)/d(pts_i) = -(pts_i - pts_j)/dist when active
            grad_pts[i] -= w * 2.0 * diff_ij
            grad_pts[j] += w * 2.0 * diff_ij
            grad_t += w * 2.0
    
    # Wall
    d_left = pts[:, 0] + HALF
    d_right = HALF - pts[:, 0]
    d_bottom = pts[:, 1] + HALF
    d_top = HALF - pts[:, 1]
    d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
    wall_viol = np.maximum(0.0, t - d_wall)
    
    if np.any(wall_viol > 0):
        max_v = np.max(wall_viol)
        exp_v = np.exp(wall_viol - max_v)
        weights = exp_v / np.sum(exp_v) / beta
        
        for i in range(N):
            w = weights[i]
            grad_t += w
            if abs(d_left[i] - d_wall[i]) < 1e-10:
                grad_pts[i, 0] += w
            if abs(d_right[i] - d_wall[i]) < 1e-10:
                grad_pts[i, 0] -= w
            if abs(d_bottom[i] - d_wall[i]) < 1e-10:
                grad_pts[i, 1] += w
            if abs(d_top[i] - d_wall[i]) < 1e-10:
                grad_pts[i, 1] -= w
    
    return np.concatenate([grad_pts.ravel(), [grad_t]])


def _violation(pts, r_target):
    """Total violation."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(N, k=1)
    pair_viol = np.maximum(0.0, 2.0 * r_target - dist[iu])
    
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    wall_viol = np.maximum(0.0, r_target - d)
    
    return float(np.sum(pair_viol) + np.sum(wall_viol))


def _simulated_annealing(pts0, r_target, deadline, rng, n_steps=2000):
    """SA to minimize violation for target radius."""
    pts = pts0.copy()
    current_viol = _violation(pts, r_target)
    best_pts = pts.copy()
    best_viol = current_viol
    
    t_start = 0.03
    t_end = 0.0001
    cooling = (t_end / t_start) ** (1.0 / n_steps)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        temp = t_start * (cooling ** step)
        n_move = rng.integers(1, 4)
        indices = rng.choice(N, size=n_move, replace=False)
        deltas = rng.normal(0, temp, size=(n_move, 2))
        
        new_pts = pts.copy()
        new_pts[indices] += deltas
        new_pts = np.clip(new_pts, -HALF, HALF)
        
        new_viol = _violation(new_pts, r_target)
        delta = new_viol - current_viol
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-10)):
            pts = new_pts
            current_viol = new_viol
            if current_viol < best_viol:
                best_viol = current_viol
                best_pts = pts.copy()
    
    return best_pts, best_viol


def _lbfgs_polish(pts0, r_target, deadline, beta=500.0, maxiter=400):
    """L-BFGS-B polish on soft-min objective for fixed target radius."""
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        diff = p[:, None, :] - p[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        iu = np.triu_indices(N, k=1)
        pair_dist = dist[iu]
        pair_viol = np.maximum(0.0, 2.0 * r_target - pair_dist)
        if np.any(pair_viol > 0):
            max_v = np.max(pair_viol)
            lse = max_v + np.log(np.sum(np.exp(pair_viol - max_v)))
            soft_pair = lse / beta
        else:
            soft_pair = 0.0
        
        d_wall = np.minimum(np.minimum(HALF - p[:, 0], HALF - p[:, 1]),
                            np.minimum(HALF + p[:, 0], HALF + p[:, 1]))
        wall_viol = np.maximum(0.0, r_target - d_wall)
        if np.any(wall_viol > 0):
            max_v = np.max(wall_viol)
            lse_w = max_v + np.log(np.sum(np.exp(wall_viol - max_v)))
            soft_wall = lse_w / beta
        else:
            soft_wall = 0.0
        
        return soft_pair + soft_wall
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        g = np.zeros_like(p)
        
        diff = p[:, None, :] - p[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        dist_safe = np.where(dist > 1e-12, dist, 1e-12)
        iu = np.triu_indices(N, k=1)
        pair_dist = dist[iu]
        pair_viol = np.maximum(0.0, 2.0 * r_target - pair_dist)
        
        if np.any(pair_viol > 0):
            max_v = np.max(pair_viol)
            exp_v = np.exp(pair_viol - max_v)
            weights = exp_v / np.sum(exp_v) / beta
            for idx in range(len(iu[0])):
                i, j = iu[0][idx], iu[1][idx]
                w = weights[idx]
                d = dist_safe[i, j]
                diff_ij = (p[i] - p[j]) / d
                g[i] -= w * 2.0 * diff_ij
                g[j] += w * 2.0 * diff_ij
        
        d_left = p[:, 0] + HALF
        d_right = HALF - p[:, 0]
        d_bottom = p[:, 1] + HALF
        d_top = HALF - p[:, 1]
        d_wall = np.minimum(np.minimum(d_left, d_right), np.minimum(d_bottom, d_top))
        wall_viol = np.maximum(0.0, r_target - d_wall)
        
        if np.any(wall_viol > 0):
            max_v = np.max(wall_viol)
            exp_v = np.exp(wall_viol - max_v)
            weights = exp_v / np.sum(exp_v) / beta
            for i in range(N):
                w = weights[i]
                if abs(d_left[i] - d_wall[i]) < 1e-10:
                    g[i, 0] += w
                if abs(d_right[i] - d_wall[i]) < 1e-10:
                    g[i, 0] -= w
                if abs(d_bottom[i] - d_wall[i]) < 1e-10:
                    g[i, 1] += w
                if abs(d_top[i] - d_wall[i]) < 1e-10:
                    g[i, 1] -= w
        
        return g.ravel()
    
    x0 = pts0.ravel()
    res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF, HALF)
    return pts


def _epigraph_optimize(pts0, t0, deadline, beta=300.0, maxiter=400):
    """Optimize jointly over positions and radius using epigraph formulation."""
    x0 = np.concatenate([pts0.ravel(), [t0]])
    
    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_min_epigraph_obj(x, beta)
    
    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        return _soft_min_epigraph_grad(x, beta)
    
    res = minimize(obj, x0, method='L-BFGS-B', jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N) + [(0.01, 0.08)],
                   options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-10})
    
    if res.x is not None:
        pts = res.x[:2 * N].reshape(N, 2)
        pts = np.clip(pts, -HALF, HALF)
        return pts
    return pts0


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice."""
    pts = []
    a = 0.12
    dx = a
    dy = a * np.sqrt(3) / 2
    row = 0
    while len(pts) < N:
        offset = (dx / 2) if (row % 2) else 0.0
        x = -HALF + a + offset
        while len(pts) < N and x < HALF - a + 1e-9:
            pts.append([x, -HALF + a + row * dy])
            x += dx
        row += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def _grid_init(rng, jitter=0.0):
    """Grid initialization."""
    m = 7
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    extra = np.array([[0.0, 0.0]])
    grid = np.vstack([grid, extra])[:N]
    if jitter > 0:
        grid = grid + rng.normal(0, jitter, size=grid.shape)
    return np.clip(grid, -HALF, HALF)


def _random_init(rng):
    """Random initialization."""
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 2))


def _spiral_init(rng, jitter=0.0):
    """Spiral initialization for diversity."""
    pts = []
    n_per_ring = [1, 6, 12, 18, 24, 30]
    total = 0
    for ring, count in enumerate(n_per_ring):
        if total >= N:
            break
        for k in range(count):
            if total >= N:
                break
            angle = 2 * np.pi * k / count + ring * 0.3
            r = 0.08 * (ring + 1)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            pts.append([x, y])
            total += 1
    pts = np.array(pts[:N])
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, size=pts.shape)
    return np.clip(pts, -HALF, HALF)


def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_r = 0.0
    
    # Phase 0: Multiple diverse initializations
    inits = [
        _hex_init(rng, jitter=0.01),
        _hex_init(rng, jitter=0.02),
        _grid_init(rng, jitter=0.01),
        _spiral_init(rng, jitter=0.01),
        _random_init(rng),
        _hex_init(rng, jitter=0.005),
    ]
    
    for init_pts in inits:
        if time.time() > deadline - 30.0:
            break
        r_init = _compute_r(init_pts)
        if r_init > best_r:
            best_r = r_init
            best_pts = init_pts.copy()
    
    if best_pts is None:
        best_pts = _hex_init(rng, jitter=0.01)
        best_r = _compute_r(best_pts)
    
    # Phase 1: Epigraph optimization with increasing beta
    beta_schedule = [100, 200, 400, 800]
    for beta in beta_schedule:
        if time.time() > deadline - 20.0:
            break
        
        # Try epigraph optimization from current best
        try:
            pts_new = _epigraph_optimize(best_pts, best_r, deadline - 20.0, beta=beta, maxiter=300)
            r_new = _compute_r(pts_new)
            if r_new > best_r:
                best_r = r_new
                best_pts = pts_new.copy()
        except Exception:
            pass
        
        # Also try from a perturbed version
        if time.time() < deadline - 15.0:
            pts_pert = best_pts + rng.normal(0, 0.005, size=(N, 2))
            pts_pert = np.clip(pts_pert, -HALF, HALF)
            try:
                pts_new2 = _epigraph_optimize(pts_pert, best_r, deadline - 15.0, beta=beta, maxiter=200)
                r_new2 = _compute_r(pts_new2)
                if r_new2 > best_r:
                    best_r = r_new2
                    best_pts = pts_new2.copy()
            except Exception:
                pass
    
    # Phase 2: SA + L-BFGS for targeted radius improvement
    for r_target in [0.0710, 0.0712, 0.0713, 0.07135]:
        if time.time() > deadline - 15.0:
            break
        
        # SA to get into basin
        pts_sa, viol_sa = _simulated_annealing(best_pts, r_target, deadline - 15.0, rng, n_steps=1500)
        
        # L-BFGS polish
        try:
            pts_lbfgs = _lbfgs_polish(pts_sa, r_target, deadline - 15.0, beta=500.0, maxiter=400)
            r_lbfgs = _compute_r(pts_lbfgs)
            if r_lbfgs > best_r:
                best_r = r_lbfgs
                best_pts = pts_lbfgs.copy()
        except Exception:
            pass
        
        # Also try from best_pts directly
        if time.time() < deadline - 10.0:
            try:
                pts_lbfgs2 = _lbfgs_polish(best_pts, r_target, deadline - 10.0, beta=500.0, maxiter=300)
                r_lbfgs2 = _compute_r(pts_lbfgs2)
                if r_lbfgs2 > best_r:
                    best_r = r_lbfgs2
                    best_pts = pts_lbfgs2.copy()
            except Exception:
                pass
    
    # Phase 3: Binary search near best radius with SA + L-BFGS
    if time.time() < deadline - 10.0:
        r_lo = best_r
        r_hi = 0.07138
        
        for _ in range(6):
            if time.time() > deadline - 5.0:
                break
            
            r_mid = (r_lo + r_hi) / 2.0
            
            pts_sa, _ = _simulated_annealing(best_pts, r_mid, deadline - 5.0, rng, n_steps=1000)
            try:
                pts_lbfgs = _lbfgs_polish(pts_sa, r_mid, deadline - 5.0, beta=800.0, maxiter=300)
                r_achieved = _compute_r(pts_lbfgs)
                if r_achieved >= r_mid - 1e-9:
                    r_lo = r_mid
                    if r_achieved > best_r:
                        best_r = r_achieved
                        best_pts = pts_lbfgs.copy()
                else:
                    r_hi = r_mid
            except Exception:
                r_hi = r_mid
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 3.0:
        # Very tight L-BFGS
        try:
            pts_final = _lbfgs_polish(best_pts, best_r * 1.000001, deadline - 3.0, beta=2000.0, maxiter=500)
            r_final = _compute_r(pts_final)
            if r_final > best_r:
                best_r = r_final
                best_pts = pts_final.copy()
        except Exception:
            pass
        
        # Epigraph with very high beta
        if time.time() < deadline - 2.0:
            try:
                pts_ep = _epigraph_optimize(best_pts, best_r, deadline - 2.0, beta=2000.0, maxiter=300)
                r_ep = _compute_r(pts_ep)
                if r_ep > best_r:
                    best_r = r_ep
                    best_pts = pts_ep.copy()
            except Exception:
                pass
    
    return best_pts
