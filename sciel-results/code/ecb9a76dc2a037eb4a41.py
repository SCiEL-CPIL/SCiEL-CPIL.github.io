"""Sphere covering with 50 points - optimized algorithm."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _covering_radius(points):
    """Compute covering radius in radians via convex-hull facet circumcenters."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

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
    Compute the smooth covering radius and its gradient w.r.t. points.
    Uses log-sum-exp over facet circumradii for a smooth surrogate.
    Returns (value, gradient) where gradient has shape (N, 3).
    """
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    try:
        hull = ConvexHull(points)
    except Exception:
        from scipy.spatial import cKDTree
        rng = np.random.RandomState(0)
        tree = cKDTree(points)
        n_samples = 30000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        ang_dists = 2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))
        m = np.max(ang_dists)
        if m < 1e-12:
            return 0.0, np.zeros_like(points)
        val = m + np.log(np.sum(np.exp(beta * (ang_dists - m)))) / beta
        return val, np.zeros_like(points)

    radii = []
    centers = []
    simplices = []
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
        radii.append(r)
        centers.append(v)
        simplices.append(simplex)

    if not radii:
        return 0.0, np.zeros_like(points)
    radii = np.array(radii)
    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    val = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta

    w = np.exp(beta * (radii - m))
    w_sum = w.sum()
    if w_sum < 1e-30:
        return val, np.zeros_like(points)
    w /= w_sum

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

def _lbfgs_refine(points, max_time=30.0, start_time=None, beta=50.0):
    """
    Refine points using L-BFGS-B on a smooth log-sum-exp surrogate for the covering radius.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)

    if start_time is None:
        start_time = time.time()

    def objective_and_grad(x_flat):
        if time.time() - start_time > max_time:
            raise Exception("timeout")
        x = x_flat.reshape(-1, 3)
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 800}
        )
        x_opt = result.x.reshape(-1, 3)
        x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    return best, best_cr

def _spherical_gd(points, max_iters=2000, lr_init=0.008, lr_min=1e-9,
                  time_limit=None, start_time=None):
    """Spherical gradient descent with adaptive step size."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    best = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    no_improve = 0

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        facets = _fast_facets(points)
        if not facets:
            break
        facets.sort(key=lambda x: -x[0])
        max_r = facets[0][0]
        K = min(10, len(facets))
        radii = np.array([f[0] for f in facets[:K]])
        beta = 40.0
        w = np.exp(beta * (radii - max_r))
        w_sum = w.sum()
        if w_sum < 1e-30:
            break
        w /= w_sum
        grad = np.zeros_like(points)
        for i in range(K):
            r, center, simplex = facets[i]
            for idx in simplex:
                p = points[idx]
                cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
                sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
                if sin_r < 1e-10:
                    continue
                g = -center / sin_r
                g_tan = g - np.dot(g, p) * p
                grad[idx] += w[i] * g_tan
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        grad /= gn
        accepted = False
        for trial in range(5):
            cand = points + lr * grad
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
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
    """
    Lloyd-style: each point moves toward the average circumcenter of its adjacent facets.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
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
        blended /= np.linalg.norm(blended, axis=1, keepdims=True)
        cr = _covering_radius(blended)
        if cr < best_cr:
            points = blended
            best = points.copy()
            best_cr = cr
        else:
            alpha = 0.2
            blended = (1 - alpha) * points + alpha * new_points
            blended /= np.linalg.norm(blended, axis=1, keepdims=True)
            cr = _covering_radius(blended)
            if cr < best_cr:
                points = blended
                best = points.copy()
                best_cr = cr
            else:
                break
    return best, best_cr

def _repulsion_refine(points, max_iters=500, time_limit=None, start_time=None):
    """
    Repulsion-based refinement: each point is pushed away from nearby points
    and attracted toward the circumcenters of its largest adjacent facets.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    best = points.copy()
    best_cr = _covering_radius(points)

    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        facets = _fast_facets(points)
        if not facets:
            break
        # Compute repulsion forces
        force = np.zeros_like(points)
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = 1.0 - np.dot(points[i], points[j])
                if d < 0.3:  # Only repel when close
                    f = (points[i] - points[j]) / max(d, 1e-6)
                    force[i] += f
                    force[j] -= f
        # Compute attraction toward facet circumcenters
        centers_per_point = [[] for _ in range(len(points))]
        for r, center, simplex in facets:
            for idx in simplex:
                centers_per_point[idx].append((r, center))
        for i in range(len(points)):
            if centers_per_point[i]:
                # Weight by radius - larger facets pull harder
                total_w = 0.0
                weighted_center = np.zeros(3)
                for r, c in centers_per_point[i]:
                    w = r ** 2
                    weighted_center += w * c
                    total_w += w
                if total_w > 1e-12:
                    target = weighted_center / total_w
                    target /= np.linalg.norm(target)
                    force[i] += 0.5 * (target - points[i])
        # Normalize and apply
        fn = np.linalg.norm(force)
        if fn < 1e-14:
            break
        force /= fn
        step = 0.02
        cand = points + step * force
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)
        cr = _covering_radius(cand)
        if cr < best_cr:
            points = cand
            best = points.copy()
            best_cr = cr
        else:
            step *= 0.5
            if step < 1e-6:
                break
            cand = points + step * force
            cand /= np.linalg.norm(cand, axis=1, keepdims=True)
            cr = _covering_radius(cand)
            if cr < best_cr:
                points = cand
                best = points.copy()
                best_cr = cr
            else:
                break
    return best, best_cr

def _simulated_annealing(points, time_limit=None, start_time=None, seed=42,
                         T_init=0.06, T_min=1e-6, cooling=0.997):
    """Simulated annealing with random perturbations of a few points."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
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

def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement with beta cascade."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms

    # Stage 1: Lloyd
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, _covering_radius(points)
        t1 = min(rem * 0.2, 5)
    else:
        t1 = 5
    points, cr = _spherical_lloyd(points, max_iters=80, time_limit=t1, start_time=start_time)

    # Stage 2: L-BFGS with low beta for broad search
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 3:
            return points, cr
        t2 = min(rem * 0.3, 15)
    else:
        t2 = 15
    points, cr = _lbfgs_refine(points, max_time=t2, start_time=start_time, beta=15.0)

    # Stage 3: Repulsion refine
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr
        t3 = min(rem * 0.2, 5)
    else:
        t3 = 5
    points, cr = _repulsion_refine(points, max_iters=300, time_limit=t3, start_time=start_time)

    # Stage 4: L-BFGS with medium beta
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 3:
            return points, cr
        t4 = min(rem * 0.25, 12)
    else:
        t4 = 12
    points, cr = _lbfgs_refine(points, max_time=t4, start_time=start_time, beta=50.0)

    # Stage 5: Lloyd again
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 2:
            return points, cr
        t5 = min(rem * 0.15, 4)
    else:
        t5 = 4
    points, cr = _spherical_lloyd(points, max_iters=60, time_limit=t5, start_time=start_time)

    # Stage 6: L-BFGS with high beta for precision
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 3:
            return points, cr
        t6 = min(rem * 0.3, 15)
    else:
        t6 = 15
    points, cr = _lbfgs_refine(points, max_time=t6, start_time=start_time, beta=150.0)

    # Stage 7: Final GD polish
    if time_limit is not None and start_time is not None:
        rem = time_limit - (time.time() - start_time)
        if rem < 1:
            return points, cr
        t7 = max(rem * 0.4, 1.0)
    else:
        t7 = 5
    points, cr = _spherical_gd(points, max_iters=1000, lr_init=0.003,
                               time_limit=t7, start_time=start_time)
    return points, cr

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=7),
        _random_points(N, seed=13),
        _icosahedron_based_points(N),
        _random_points(N, seed=99),
    ]

    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)

    # Phase 1: Refine each initial config with a time budget
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 8:
            break
        time_per = min(remaining / (len(initial_configs) - i), 20.0)
        refined, cr = _refine(init_pts, time_limit=time_per, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()

    # Phase 2: Simulated annealing from best
    remaining = deadline - time.time()
    if remaining > 12:
        sa_time = min(remaining * 0.3, 18.0)
        sa_pts, sa_cr = _simulated_annealing(best_points, time_limit=sa_time,
                                             start_time=start_time, seed=777)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_pts.copy()

    # Phase 3: Final refinement
    remaining = deadline - time.time()
    if remaining > 8:
        fin_time = min(remaining * 0.5, 20.0)
        fin_pts, fin_cr = _refine(best_points, time_limit=fin_time, start_time=start_time)
        if fin_cr < best_cr:
            best_cr = fin_cr
            best_points = fin_pts.copy()

    # Final L-BFGS polish with very high beta
    remaining = deadline - time.time()
    if remaining > 5:
        lbfgs_time = max(remaining * 0.7, 3.0)
        lbfgs_pts, lbfgs_cr = _lbfgs_refine(best_points, max_time=lbfgs_time,
                                            start_time=start_time, beta=300.0)
        if lbfgs_cr < best_cr:
            best_cr = lbfgs_cr
            best_points = lbfgs_pts.copy()

    # Final GD micro-polish
    remaining = deadline - time.time()
    if remaining > 2:
        gd_time = max(remaining * 0.8, 1.0)
        gd_pts, gd_cr = _spherical_gd(best_points, max_iters=500, lr_init=0.002,
                                      time_limit=gd_time, start_time=start_time)
        if gd_cr < best_cr:
            best_cr = gd_cr
            best_points = gd_pts.copy()

    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    return best_points
# EVOLVE-BLOCK-END
