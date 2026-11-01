"""Pack 200 equal circles in a unit square via soft-contact dynamics + gradient polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 150.0
MARGIN = 15.0


def _score(pts):
    """Largest common radius for a set of centres (raw value)."""
    if pts is None or len(pts) != N or pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    wall_min = float(np.min(wall))
    if wall_min <= 0:
        return 0.0
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair_min = float(np.min(d)) / 2.0
    return min(wall_min, pair_min)


def _soft_min(pts, lam):
    """Compute soft-min of all distance terms (pairwise/2 and wall distances)."""
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    terms = np.concatenate([wall, pair.ravel()])
    terms = np.clip(terms, -1e6, 1e6)
    m = np.min(terms)
    z = np.exp(-lam * (terms - m))
    r_soft = m + (1.0 / lam) * np.log(np.sum(z))
    return r_soft


def _soft_min_neg(pts, lam):
    """Objective to minimize: -soft_min."""
    return -_soft_min(pts, lam)


def _soft_min_grad(pts, lam):
    """Gradient of -soft_min wrt pts."""
    pts = np.asarray(pts)
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    terms = np.concatenate([wall, pair.ravel()])
    terms = np.clip(terms, -1e6, 1e6)
    m = np.min(terms)
    e = np.exp(-lam * (terms - m))
    w = e / np.sum(e)

    # Wall gradient
    g_wall = np.zeros_like(pts)
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    mx = ax > ay
    g_wall[mx, 0] = -np.sign(pts[mx, 0])
    my = ay > ax
    g_wall[my, 1] = -np.sign(pts[my, 1])
    tie = (ax == ay) & (ax > 0)
    g_wall[tie, 0] = -0.5 * np.sign(pts[tie, 0])
    g_wall[tie, 1] = -0.5 * np.sign(pts[tie, 1])

    # Pair gradient
    w_pair = w[N:].reshape(N, N)
    w_pair = w_pair * (1.0 - np.eye(N))
    diff = pts[:, None, :] - pts[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dmat, np.inf)
    factor = w_pair / (2.0 * dmat)
    g_pair = (factor[..., None] * diff).sum(axis=1)

    g_terms = g_wall + g_pair
    return -g_terms


def _hex_init():
    """Hexagonal lattice initialization."""
    best = None
    best_score = 0
    for s in np.linspace(0.035, 0.085, 300):
        vstep = s * np.sqrt(3) / 2.0
        pts = []
        x = -0.5 + s / 2.0
        row = 0
        y = -0.5 + vstep / 2.0
        while y <= 0.5:
            offset = (s / 2.0) if (row % 2) else 0.0
            xx = x + offset
            while xx <= 0.5:
                pts.append((xx, y))
                xx += s
            y += vstep
            row += 1
        pts = np.array(pts)
        if len(pts) >= N:
            r2 = (pts ** 2).sum(1)
            idx = np.argsort(r2)
            pts = pts[idx[:N]]
            sc = _score(pts)
            if sc > 0 and sc > best_score:
                best = pts.copy()
                best_score = sc
    if best is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    return best


def _dynamics_relax(pts, t_budget=40.0):
    """
    Soft-contact dynamics: circles repel via polynomial potential,
    walls repel via polynomial potential. Symplectic Euler with adaptive dt.
    """
    pts = pts.astype(np.float64).copy()
    vel = np.zeros_like(pts)
    n = len(pts)

    # Potential parameters
    # Repulsion: V = k * max(0, overlap)^p where overlap = target_r - dist
    # We'll use a fixed target radius that we'll tune
    # Start with a reasonable target
    r_target = _score(pts) * 1.001  # slightly above current
    k_rep = 50.0
    p_rep = 4  # high-order polynomial
    k_wall = 50.0
    p_wall = 4

    # Damping
    damping = 0.95

    # Adaptive time step
    dt = 0.002
    dt_min = 1e-6
    dt_max = 0.01

    t0 = time.time()
    max_time = t_budget

    for step in range(50000):
        if time.time() - t0 > max_time:
            break

        # Compute forces
        forces = np.zeros_like(pts)

        # Wall forces: push inward if too close to wall
        # wall_dist_i = 0.5 - max(|x_i|, |y_i|)
        # If wall_dist < r_target, push inward
        ax = np.abs(pts[:, 0])
        ay = np.abs(pts[:, 1])
        wall_dist = 0.5 - np.maximum(ax, ay)

        for i in range(n):
            # Determine which wall is closest
            if ax[i] >= ay[i]:
                # x-wall
                if wall_dist[i] < r_target:
                    overlap = r_target - wall_dist[i]
                    f = k_rep * overlap ** p_rep
                    forces[i, 0] += -np.sign(pts[i, 0]) * f
            else:
                # y-wall
                if wall_dist[i] < r_target:
                    overlap = r_target - wall_dist[i]
                    f = k_rep * overlap ** p_rep
                    forces[i, 1] += -np.sign(pts[i, 1]) * f

        # Pairwise repulsion
        # Vectorized: compute all pairwise distances
        diff = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(dist, np.inf)

        # Overlap: r_target - dist, only where positive
        overlap = np.maximum(0, r_target - dist)
        # Force magnitude: k * overlap^p
        f_mag = k_rep * overlap ** p_rep
        # Direction: diff/dist (unit vector from j to i)
        # Avoid division by zero
        safe_dist = np.where(dist > 1e-10, dist, 1.0)
        unit = diff / safe_dist[..., None]
        # Force on i from j: f_mag[i,j] * unit[i,j] (pushes i away from j)
        # Sum over j
        forces += (f_mag[..., None] * unit).sum(axis=1)

        # Symplectic Euler: v += F*dt, x += v*dt
        vel = vel * damping + forces * dt
        pts = pts + vel * dt

        # Clamp to square (hard constraint)
        pts = np.clip(pts, -0.5, 0.5)

        # Adaptive time step: reduce if velocities are large
        v_max = np.max(np.abs(vel))
        if v_max > 0.1:
            dt = max(dt_min, dt * 0.5)
        elif v_max < 0.01:
            dt = min(dt_max, dt * 1.1)

        # Check convergence
        if v_max < 1e-5 and step > 100:
            break

    return pts


def _gradient_polish(pts, t_budget=30.0):
    """Polish with L-BFGS-B using soft-min objective."""
    pts = pts.astype(np.float64).copy()
    lo = np.full(N * 2, -0.5)
    hi = np.full(N * 2, 0.5)

    t0 = time.time()
    max_time = t_budget

    # Anneal lambda
    lam_schedule = [100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]

    cur = pts.copy()
    for lam in lam_schedule:
        if time.time() - t0 > max_time:
            break

        def f(x):
            return _soft_min_neg(x.reshape(N, 2), lam)

        def g(x):
            return _soft_min_grad(x.reshape(N, 2), lam).ravel()

        try:
            res = minimize(
                f, cur.ravel(), jac=g, method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": 500, "maxfun": 3000, "ftol": 1e-12, "gtol": 1e-10},
            )
            cand = res.x.reshape(N, 2)
            cand = np.clip(cand, -0.5, 0.5)
            if _score(cand) >= _score(cur):
                cur = cand
        except Exception:
            pass

    return cur


def _slsqp_active_polish(pts, t_budget=20.0):
    """SLSQP polish focusing on active constraints."""
    pts = pts.astype(np.float64).copy()
    t0 = time.time()
    max_time = t_budget

    r = _score(pts)
    if r <= 0:
        return pts

    lo = np.full(N * 2, -0.5)
    hi = np.full(N * 2, 0.5)

    # Use SLSQP with inequality constraints: dist_ij >= 2r, wall_dist_i >= r
    # But this is expensive for 200 points. Instead, do a few rounds of
    # L-BFGS-B with very sharp lambda.
    cur = pts.copy()
    for lam in [5000.0, 10000.0, 20000.0]:
        if time.time() - t0 > max_time:
            break

        def f(x):
            return _soft_min_neg(x.reshape(N, 2), lam)

        def g(x):
            return _soft_min_grad(x.reshape(N, 2), lam).ravel()

        try:
            res = minimize(
                f, cur.ravel(), jac=g, method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": 800, "maxfun": 5000, "ftol": 1e-14, "gtol": 1e-12},
            )
            cand = res.x.reshape(N, 2)
            cand = np.clip(cand, -0.5, 0.5)
            if _score(cand) >= _score(cur):
                cur = cand
        except Exception:
            pass

    return cur


def construct():
    t0 = time.time()
    limit = DEADLINE - MARGIN

    # Phase 1: Hexagonal initialization
    pts = _hex_init().astype(np.float64)
    best = pts.copy()
    best_score = _score(best)

    # Phase 2: Dynamics relaxation
    if time.time() - t0 < limit:
        t_dyn = min(40.0, limit - (time.time() - t0) - 5.0)
        if t_dyn > 5.0:
            dyn_pts = _dynamics_relax(best, t_budget=t_dyn)
            dyn_pts = np.clip(dyn_pts, -0.5, 0.5)
            dyn_score = _score(dyn_pts)
            if dyn_score > best_score:
                best = dyn_pts.copy()
                best_score = dyn_score

    # Phase 3: Gradient polish
    if time.time() - t0 < limit:
        t_pol = min(30.0, limit - (time.time() - t0) - 5.0)
        if t_pol > 5.0:
            pol_pts = _gradient_polish(best, t_budget=t_pol)
            pol_pts = np.clip(pol_pts, -0.5, 0.5)
            pol_score = _score(pol_pts)
            if pol_score > best_score:
                best = pol_pts.copy()
                best_score = pol_score

    # Phase 4: Final active-constraint polish
    if time.time() - t0 < limit:
        t_act = min(15.0, limit - (time.time() - t0) - 2.0)
        if t_act > 3.0:
            act_pts = _slsqp_active_polish(best, t_budget=t_act)
            act_pts = np.clip(act_pts, -0.5, 0.5)
            act_score = _score(act_pts)
            if act_score > best_score:
                best = act_pts.copy()
                best_score = act_score

    # Final clamp
    best = np.clip(best, -0.5, 0.5)
    return best


if __name__ == "__main__":
    t = time.time()
    p = construct()
    print("elapsed", time.time() - t, "score", _score(p), "shape", p.shape)
