"""Rewritten program for sphere covering with 50 points."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    """Normalize points to unit sphere."""
    pts = np.asarray(pts, dtype=float)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(points):
    """Compute covering radius in radians via convex-hull facet circumcenters."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(points)
        n_samples = 50000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return float(np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))))

    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    return float(max_r)

def _fast_facets(points):
    """Return list of (radius, circumcenter, facet_indices) for triangular facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        out.append((r, v, simplex))
    return out

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute the smooth covering radius (log-sum-exp over facet radii) and its
    gradient with respect to the input points.
    """
    points = _normalize(points)
    facets = _fast_facets(points)
    if not facets:
        return 0.0, np.zeros_like(points)

    radii = np.array([f[0] for f in facets])
    centers = np.array([f[1] for f in facets])
    simplices = [f[2] for f in facets]

    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    val = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta

    w = np.exp(beta * (radii - m))
    w /= w.sum()

    grad = np.zeros_like(points)
    for i in range(len(radii)):
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan

    return val, grad

def _riesz_energy_grad(points, s=2.0):
    """
    Compute Riesz s-energy and its gradient. Lower energy = better spread.
    This provides a smooth auxiliary objective for point distribution.
    """
    points = _normalize(points)
    n = len(points)
    energy = 0.0
    grad = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-12:
                continue
            energy += 1.0 / (d ** s)
            diff = points[i] - points[j]
            f_mag = s / (d ** (s + 1))
            f_dir = diff / d
            # Project to tangent space
            f_tan_i = f_dir - np.dot(f_dir, points[i]) * points[i]
            f_tan_j = -f_dir + np.dot(f_dir, points[j]) * points[j]
            grad[i] += f_mag * f_tan_i
            grad[j] += f_mag * f_tan_j
    
    return energy, grad

def _lbfgs_refine(points, max_time=30.0, start_time=None, beta=50.0):
    """Refine points using L-BFGS-B on a smooth log-sum-exp surrogate."""
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 1000}
    )

    x_opt = result.x.reshape(-1, 3)
    x_opt = _normalize(x_opt)
    cr_opt = _covering_radius(x_opt)

    if cr_opt < best_cr:
        best = x_opt.copy()
        best_cr = cr_opt

    return best, best_cr

def _lbfgs_riesz(points, max_time=20.0, start_time=None, s=2.0, weight=0.1):
    """
    Refine using combined LSE covering radius + Riesz energy objective.
    The Riesz term helps maintain good distribution while LSE minimizes max radius.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        lse_val, lse_grad = _smooth_covering_radius_grad(x, beta=50.0)
        riesz_val, riesz_grad = _riesz_energy_grad(x, s=s)
        val = lse_val + weight * riesz_val
        grad = lse_grad + weight * riesz_grad
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800}
    )

    x_opt = result.x.reshape(-1, 3)
    x_opt = _normalize(x_opt)
    cr_opt = _covering_radius(x_opt)

    if cr_opt < best_cr:
        best = x_opt.copy()
        best_cr = cr_opt

    return best, best_cr

def _spherical_gd(points, max_iters=2000, lr_init=0.008, lr_min=1e-9,
                  time_limit=None, start_time=None):
    """Spherical gradient descent with adaptive step size."""
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    no_improve = 0

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        val, grad = _smooth_covering_radius_grad(points, beta=30.0)
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        grad /= gn
        accepted = False
        for trial in range(5):
            cand = points + lr * grad
            cand = _normalize(cand)
            cand_cr = _covering_radius(cand)
            if cand_cr < best_cr - 1e-14:
                points = cand
                best = cand.copy()
                best_cr = cand_cr
                lr = min(lr * 1.05, 0.05)
                no_improve = 0
                accepted = True
                break
            lr = lr * 0.7
        if not accepted:
            no_improve += 1
            lr = max(lr * 0.85, lr_min)
            if no_improve > 80 or lr < lr_min * 1.01:
                break
    return best, best_cr

def _spherical_lloyd(points, max_iters=200, time_limit=None, start_time=None):
    """Lloyd-style: each point moves to the average circumcenter of adjacent facets."""
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        facets = _fast_facets(points)
        if not facets:
            break
        centers_per_point = [[] for _ in range(len(points))]
        for r, center, simplex in facets:
            for idx in simplex:
                centers_per_point[idx].append(center)
        new_points = np.zeros_like(points)
        for i in range(len(points)):
            if not centers_per_point[i]:
                new_points[i] = points[i]
            else:
                c = np.mean(centers_per_point[i], axis=0)
                cn = np.linalg.norm(c)
                if cn < 1e-12:
                    new_points[i] = points[i]
                else:
                    new_points[i] = c / cn
        alpha = 0.5
        blended = (1 - alpha) * points + alpha * new_points
        blended = _normalize(blended)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended = _normalize(blended)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break
    return best, best_cr

def _simulated_annealing(points, time_limit=None, start_time=None, seed=42,
                         T_init=0.06, T_min=1e-6, cooling=0.997):
    """Simulated annealing with random perturbations."""
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    rng = np.random.RandomState(seed)
    T = T_init
    cur = points.copy()
    cur_cr = best_cr

    while T > T_min:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        n_pert = rng.randint(1, 4)
        idxs = rng.choice(N, size=n_pert, replace=False)
        cand = cur.copy()
        for i in idxs:
            noise = rng.randn(3) * T
            cand[i] = cur[i] + noise
            cand[i] /= np.linalg.norm(cand[i])
        cand_cr = _covering_radius(cand)
        delta = cand_cr - cur_cr
        if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-12)):
            cur = cand
            cur_cr = cand_cr
            if cand_cr < best_cr:
                best = cand.copy()
                best_cr = cand_cr
        T *= cooling
    return best, best_cr

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _jittered_fibonacci_points(n, seed=42, jitter=0.02):
    """Fibonacci points with small random jitter to break symmetry."""
    pts = _fibonacci_points(n)
    rng = np.random.RandomState(seed)
    for i in range(n):
        noise = rng.randn(3) * jitter
        pts[i] = pts[i] + noise
        pts[i] /= np.linalg.norm(pts[i])
    return pts

def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _icosahedron_based_points(n):
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mn = np.linalg.norm(mid)
        if mn > 1e-10:
            mid /= mn
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    return pts[:n]

def _repulsion_refine(points, time_limit=None, start_time=None,
                      n_iters=500, dt=0.01, alpha=4.0):
    """
    Symplectic velocity Verlet with inverse-power repulsion.
    Points repel each other with force ~ 1/d^alpha, projected to tangent space.
    """
    points = _normalize(points)
    n = len(points)

    velocities = np.zeros_like(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    for it in range(n_iters):
        if time_limit is not None and time.time() - start_time > time_limit:
            break

        # Compute forces (repulsion)
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = points[i] - points[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    continue
                f_mag = 1.0 / (dist ** alpha)
                # Project to tangent space of point i
                f_dir = d / dist
                f_tan = f_dir - np.dot(f_dir, points[i]) * points[i]
                forces[i] += f_mag * f_tan

        # Velocity Verlet update
        points_new = points + velocities * dt + 0.5 * forces * dt * dt
        points_new = _normalize(points_new)

        # Compute new forces
        forces_new = np.zeros_like(points)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = points_new[i] - points_new[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    continue
                f_mag = 1.0 / (dist ** alpha)
                f_dir = d / dist
                f_tan = f_dir - np.dot(f_dir, points_new[i]) * points_new[i]
                forces_new[i] += f_mag * f_tan

        velocities = 0.5 * (forces + forces_new) * dt
        velocities /= np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-15
        # Damping
        velocities *= 0.99

        points = points_new

        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best = points.copy()

    return best, best_cr

def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: Lloyd -> L-BFGS -> Lloyd -> L-BFGS -> Riesz -> GD."""
    points = _normalize(points)

    if time_limit is None:
        time_limit = 60.0
    if start_time is None:
        start_time = time.time()

    # Stage 1: Lloyd
    rem = time_limit - (time.time() - start_time)
    if rem < 2:
        return points, _covering_radius(points)
    t1 = min(rem * 0.15, 4)
    points, cr = _spherical_lloyd(points, max_iters=100, time_limit=t1, start_time=start_time)

    # Stage 2: L-BFGS with smooth surrogate
    rem = time_limit - (time.time() - start_time)
    if rem < 3:
        return points, cr
    t2 = min(rem * 0.3, 18)
    points, cr = _lbfgs_refine(points, max_time=t2, start_time=start_time, beta=50.0)

    # Stage 3: Lloyd again
    rem = time_limit - (time.time() - start_time)
    if rem < 2:
        return points, cr
    t3 = min(rem * 0.12, 3)
    points, cr = _spherical_lloyd(points, max_iters=80, time_limit=t3, start_time=start_time)

    # Stage 4: Riesz-assisted L-BFGS
    rem = time_limit - (time.time() - start_time)
    if rem < 3:
        return points, cr
    t4 = min(rem * 0.25, 12)
    points, cr = _lbfgs_riesz(points, max_time=t4, start_time=start_time, s=2.0, weight=0.05)

    # Stage 5: L-BFGS polish with higher beta
    rem = time_limit - (time.time() - start_time)
    if rem < 3:
        return points, cr
    t5 = min(rem * 0.25, 10)
    points, cr = _lbfgs_refine(points, max_time=t5, start_time=start_time, beta=100.0)

    # Stage 6: Final GD polish
    rem = time_limit - (time.time() - start_time)
    if rem < 1:
        return points, cr
    t6 = max(rem * 0.5, 1.0)
    points, cr = _spherical_gd(points, max_iters=1000, lr_init=0.004,
                               time_limit=t6, start_time=start_time)
    return points, cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    initial_configs = [
        _fibonacci_points(N),
        _jittered_fibonacci_points(N, seed=42, jitter=0.015),
        _random_points(N, seed=1),
        _random_points(N, seed=7),
        _random_points(N, seed=13),
        _icosahedron_based_points(N),
        _random_points(N, seed=99),
        _jittered_fibonacci_points(N, seed=123, jitter=0.03),
    ]

    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)

    # Phase 1: Repulsion-based refinement on each initial config
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_per = min(remaining / (len(initial_configs) - i), 10.0)
        rep_pts, rep_cr = _repulsion_refine(init_pts, time_limit=time_per, start_time=start_time)
        if rep_cr < best_cr:
            best_cr = rep_cr
            best_points = rep_pts.copy()

    # Phase 2: Full multi-stage refinement on best
    remaining = deadline - time.time()
    if remaining > 10:
        refine_time = min(remaining * 0.45, 35.0)
        refined, cr = _refine(best_points, time_limit=refine_time, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()

    # Phase 3: Simulated annealing
    remaining = deadline - time.time()
    if remaining > 12:
        sa_time = min(remaining * 0.2, 12.0)
        sa_pts, sa_cr = _simulated_annealing(best_points, time_limit=sa_time,
                                             start_time=start_time, seed=777)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_pts.copy()

    # Phase 4: Final refinement
    remaining = deadline - time.time()
    if remaining > 8:
        fin_time = min(remaining * 0.4, 18.0)
        fin_pts, fin_cr = _refine(best_points, time_limit=fin_time, start_time=start_time)
        if fin_cr < best_cr:
            best_cr = fin_cr
            best_points = fin_pts.copy()

    # Phase 5: Final L-BFGS polish with very high beta
    remaining = deadline - time.time()
    if remaining > 5:
        lbfgs_time = max(remaining * 0.6, 3.0)
        lbfgs_pts, lbfgs_cr = _lbfgs_refine(best_points, max_time=lbfgs_time,
                                            start_time=start_time, beta=300.0)
        if lbfgs_cr < best_cr:
            best_cr = lbfgs_cr
            best_points = lbfgs_pts.copy()

    # Phase 6: Riesz-assisted final polish
    remaining = deadline - time.time()
    if remaining > 4:
        riesz_time = max(remaining * 0.4, 2.0)
        r_pts, r_cr = _lbfgs_riesz(best_points, max_time=riesz_time,
                                    start_time=start_time, s=2.5, weight=0.02)
        if r_cr < best_cr:
            best_cr = r_cr
            best_points = r_pts.copy()

    # Phase 7: Final GD micro-polish
    remaining = deadline - time.time()
    if remaining > 2:
        gd_time = max(remaining * 0.7, 1.0)
        gd_pts, gd_cr = _spherical_gd(best_points, max_iters=500, lr_init=0.002,
                                      time_limit=gd_time, start_time=start_time)
        if gd_cr < best_cr:
            best_cr = gd_cr
            best_points = gd_pts.copy()

    best_points = _normalize(best_points)
    return best_points
# EVOLVE-BLOCK-END
