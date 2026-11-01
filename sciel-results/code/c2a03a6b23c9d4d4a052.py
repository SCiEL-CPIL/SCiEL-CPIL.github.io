"""
Packing of 50 equal circles in a unit square.
Uses a combination of force-directed relaxation, simulated annealing, and L-BFGS optimization.
"""
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
    # boundary: min distance from each point to the square edge
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    r = float(np.min(d))
    # pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r = min(r, float(np.min(dist)) / 2.0)
    return r


def _softmin_score(pts, t):
    """Smooth objective to maximize: soft-min of all constraints."""
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0
    vals = np.concatenate([d, pair])
    m = np.min(vals)
    s = np.sum(np.exp(-t * (vals - m)))
    return m - np.log(s) / t


def _violation_score(pts, r_target, t):
    """Smooth objective: minimize total violation of radius r_target."""
    d = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                   np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist / 2.0

    boundary_viol = np.maximum(0, r_target - d)
    pair_viol = np.maximum(0, r_target - pair)

    all_viol = np.concatenate([boundary_viol, pair_viol])
    if np.max(all_viol) < 1e-15:
        return 0.0
    # Use a soft-max to make it smooth
    m = np.max(all_viol)
    s = np.sum(np.exp(t * (all_viol - m)))
    return np.log(s) / t + m - 1.0 / t


def _lbfgs_maximize(pts0, t, maxiter, deadline):
    """Run L-BFGS-B to maximize the soft-min score."""
    x0 = pts0.ravel()

    def neg_obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return -_softmin_score(p, t)

    def neg_grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        # Numerical gradient
        eps = 1e-5
        g = np.zeros_like(x)
        base = neg_obj(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            g[i] = (neg_obj(xp) - base) / eps
        return g

    res = minimize(neg_obj, x0, method='L-BFGS-B',
                   jac=neg_grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
    return pts, float(-res.fun)


def _lbfgs_violation(pts0, r_target, t, maxiter, deadline):
    """Run L-BFGS-B to minimize violation of radius r_target."""
    x0 = pts0.ravel()

    def obj(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        return _violation_score(p, r_target, t)

    def grad(x):
        if time.time() > deadline:
            raise RuntimeError("timeout")
        p = x.reshape(N, 2)
        eps = 1e-5
        g = np.zeros_like(x)
        base = obj(x)
        for i in range(x.size):
            xp = x.copy()
            xp[i] += eps
            g[i] = (obj(xp) - base) / eps
        return g

    res = minimize(obj, x0, method='L-BFGS-B',
                   jac=grad,
                   bounds=[(-HALF, HALF)] * (2 * N),
                   options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
    pts = res.x.reshape(N, 2)
    pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
    return pts, float(res.fun)


def _hex_init(rng, jitter=0.0):
    """Hexagonal close-packed lattice, jittered."""
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
    return np.clip(pts, -HALF + 1e-9, HALF - 1e-9)


def _force_relax(pts, r_target, n_steps, deadline):
    """
    Force-directed relaxation.
    Pushes overlapping circles apart and circles outside the boundary inwards.
    """
    pts = pts.copy()
    dt = 0.01
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 2 * r_target and dist > 1e-9:
                    # Repulsive force proportional to overlap
                    overlap = 2 * r_target - dist
                    force_mag = overlap / dist
                    forces[i] += diff * force_mag
                    forces[j] -= diff * force_mag
        
        # Boundary attraction (push in if too close to edge)
        for i in range(N):
            # x bounds
            if pts[i, 0] < -HALF + r_target:
                forces[i, 0] += (-HALF + r_target - pts[i, 0])
            elif pts[i, 0] > HALF - r_target:
                forces[i, 0] += (HALF - r_target - pts[i, 0])
            
            # y bounds
            if pts[i, 1] < -HALF + r_target:
                forces[i, 1] += (-HALF + r_target - pts[i, 1])
            elif pts[i, 1] > HALF - r_target:
                forces[i, 1] += (HALF - r_target - pts[i, 1])
        
        # Update positions
        pts += forces * dt
        pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)
        
        # Cool down step size
        dt *= 0.999
        
    return pts


def construct():
    start = time.time()
    deadline = start + 140.0  # Safety margin
    rng = np.random.default_rng(12345)

    best_pts = _hex_init(rng, jitter=0.01)
    best_r = _compute_r(best_pts)

    def try_update(pts):
        nonlocal best_pts, best_r
        r = _compute_r(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 1: Force relaxation with increasing target radius
    # Start with a target radius slightly larger than current best
    r_target = best_r * 1.05
    for i in range(5):
        if time.time() > deadline - 30.0:
            break
        # Relax
        relaxed_pts = _force_relax(best_pts, r_target, 200, deadline)
        try_update(relaxed_pts)
        
        # Increase target radius
        r_target = best_r * (1.05 + 0.01 * i)

    # Phase 2: L-BFGS optimization to maximize soft-min radius
    if time.time() < deadline - 40.0:
        for t in [10.0, 20.0, 50.0]:
            if time.time() > deadline - 10.0:
                break
            try:
                pts, score = _lbfgs_maximize(best_pts, t=t, maxiter=200, deadline=deadline)
                try_update(pts)
            except Exception:
                pass

    # Phase 3: Binary search on radius using violation minimization
    if time.time() < deadline - 20.0:
        r_low = best_r
        r_high = best_r * 1.1
        
        for iteration in range(10):
            if time.time() > deadline - 10.0:
                break
            r_mid = (r_low + r_high) / 2.0
            
            try:
                pts, viol = _lbfgs_violation(best_pts, r_mid, t=50.0, maxiter=200, deadline=deadline)
                try_update(pts)
            except Exception:
                pass
            
            r_achieved = _compute_r(best_pts)
            if r_achieved >= r_mid - 1e-8:
                r_low = r_mid
            else:
                r_high = r_mid

    # Phase 4: Final polish
    if time.time() < deadline - 5.0:
        for t in [100.0, 200.0]:
            if time.time() > deadline - 2.0:
                break
            try:
                pts, score = _lbfgs_maximize(best_pts, t=t, maxiter=300, deadline=deadline)
                try_update(pts)
            except Exception:
                pass

    return best_pts
