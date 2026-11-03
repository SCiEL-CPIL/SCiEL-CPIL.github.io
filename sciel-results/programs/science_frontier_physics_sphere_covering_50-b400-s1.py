"""Combined approach: multi-start scaffolding + smooth LSE covering-radius refinement + Riesz polish."""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

# ---------------- utilities ----------------

def _normalize(points):
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _covering_radius(points):
    """Exact covering radius (radians) via convex hull facet circumcenters."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: dense random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(0)
        n_samples = 60000
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


def _facet_data(points):
    """Return (radii, centers, simplices) for triangular facets."""
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.array([]), [], []

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

    return np.array(radii), centers, simplices


def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Smooth surrogate: log-sum-exp of facet circumradii.
    Returns (value, gradient) with gradient in ambient R^3 (not projected).
    """
    points = _normalize(points)
    radii, centers, simplices = _facet_data(points)
    n = len(radii)
    if n == 0:
        return 0.0, np.zeros_like(points)

    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)

    # numerically stable LSE
    exps = np.exp(beta * (radii - m))
    lse = m + np.log(np.sum(exps)) / beta
    w = exps / np.sum(exps)

    grad = np.zeros_like(points)
    for i in range(n):
        wi = w[i]
        if wi < 1e-15:
            continue
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(max(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            # project to tangent plane at p
            g_tan = g - np.dot(g, p) * p
            grad[idx] += wi * g_tan

    return float(lse), grad


def _riesz_energy_grad(points, s=12.0):
    """
    Riesz s-energy (sum over pairs of angular distance^{-s}) and gradient (tangent-projected).
    Used as an auxiliary smoother to diversify local minima.
    """
    points = _normalize(points)
    n = len(points)
    dots = points @ points.T
    dots = np.clip(dots, -1.0 + 1e-15, 1.0 - 1e-15)
    angles = np.arccos(dots)
    sins = np.sin(angles)
    sins = np.maximum(sins, 1e-15)

    ang_powers = angles ** (-s - 1)
    np.fill_diagonal(ang_powers, 0.0)
    np.fill_diagonal(sins, 1.0)
    np.fill_diagonal(dots, 1.0)

    weights = s * ang_powers / sins
    np.fill_diagonal(weights, 0.0)

    grad = weights @ points
    grad_tan = grad - np.sum(grad * points, axis=1, keepdims=True) * points

    iu = np.triu_indices(n, k=1)
    energy = float(np.sum(angles[iu] ** (-s)))
    return energy, grad_tan


def _lbfgs_smooth(points, start_time=None, deadline=None, beta=50.0, maxiter=300,
                  ftol=1e-14, gtol=1e-12, maxfun=500):
    """
    L-BFGS-B on smooth LSE surrogate.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    def obj_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        # project gradient to tangent space
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        return val, grad_tan.flatten()

    x0 = points.flatten()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
                "maxfun": maxfun,
            }
        )
        x_opt = res.x.reshape(-1, 3)
        x_opt = _normalize(x_opt)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    # time guard
    if deadline is not None and time.time() >= deadline:
        return best, best_cr

    return best, best_cr


def _lbfgs_riesz(points, start_time=None, deadline=None, s=12.0, maxiter=200,
                 ftol=1e-12, gtol=1e-10, maxfun=400):
    """
    L-BFGS-B on Riesz energy (auxiliary smoother).
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    def obj_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _normalize(x)
        val, grad = _riesz_energy_grad(x, s=s)
        # grad already tangent-projected
        return val, grad.flatten()

    x0 = points.flatten()
    try:
        res = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={
                "maxiter": maxiter,
                "ftol": ftol,
                "gtol": gtol,
                "maxfun": maxfun,
            }
        )
        x_opt = res.x.reshape(-1, 3)
        x_opt = _normalize(x_opt)
        cr_opt = _covering_radius(x_opt)
        if cr_opt < best_cr:
            best = x_opt.copy()
            best_cr = cr_opt
    except Exception:
        pass

    if deadline is not None and time.time() >= deadline:
        return best, best_cr

    return best, best_cr


def _lloyd_iteration(points, max_iters=300, tol=1e-10, start_time=None, deadline=None):
    """
    Spherical Lloyd: move each point toward the mean of circumcenters of incident facets.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)

    for _ in range(max_iters):
        if deadline is not None and time.time() >= deadline:
            break

        try:
            hull = ConvexHull(points)
        except Exception:
            break

        new_points = points.copy()
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)

        for i in range(len(points)):
            if not point_to_facets[i]:
                continue
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                vn = np.linalg.norm(v)
                if vn < 1e-12:
                    continue
                v /= vn
                d = np.dot(v, a)
                if d < 0:
                    v = -v
                circumcenters.append(v)
            if not circumcenters:
                continue
            centroid = np.mean(circumcenters, axis=0)
            cn = np.linalg.norm(centroid)
            if cn > 1e-12:
                new_points[i] = centroid / cn

        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best = points.copy()
        if diff < tol:
            break

    return best, best_cr


def _simulated_annealing(points, start_time=None, deadline=None, seed=123):
    """
    Short SA to escape local minima before final polish.
    """
    points = _normalize(points)
    best = points.copy()
    best_cr = _covering_radius(points)
    current = points.copy()
    current_cr = best_cr

    rng = np.random.RandomState(seed)
    T = 0.03
    cooling = 0.997
    T_min = 1e-5

    while T > T_min:
        if deadline is not None and time.time() >= deadline:
            break

        n_pert = rng.randint(1, 4)
        idxs = rng.choice(N, size=n_pert, replace=False)
        cand = current.copy()
        for idx in idxs:
            noise = rng.randn(3) * T
            cand[idx] = current[idx] + noise
            cand[idx] /= np.linalg.norm(cand[idx])

        new_cr = _covering_radius(cand)
        delta = new_cr - current_cr
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current = cand
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best = cand.copy()

        T *= cooling

    return best, best_cr


def _multi_beta_lse(points, start_time, deadline):
    """
    Multi-stage beta cascade with progressively tighter L-BFGS tolerances.
    """
    best = _normalize(points)
    best_cr = _covering_radius(best)

    stages = [
        (15.0, 1e-10, 1e-8, 200, 300),
        (50.0, 1e-12, 1e-10, 300, 400),
        (150.0, 1e-14, 1e-12, 300, 400),
        (300.0, 1e-15, 1e-13, 300, 400),
        (500.0, 1e-16, 1e-14, 300, 400),
    ]

    for beta, ftol, gtol, maxiter, maxfun in stages:
        if deadline is not None and time.time() >= deadline:
            break
        # time budget per stage
        remaining = max(1.0, deadline - time.time()) if deadline else 10.0
        # allocate up to 12s per stage, but not more than remaining
        stage_time = min(12.0, remaining * 0.6)

        # run L-BFGS with a local deadline
        local_deadline = time.time() + stage_time
        pts, cr = _lbfgs_smooth(
            best,
            start_time=start_time,
            deadline=local_deadline,
            beta=beta,
            maxiter=maxiter,
            ftol=ftol,
            gtol=gtol,
            maxfun=maxfun,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

        # optional Riesz polish at lower beta to diversify
        if deadline is not None and time.time() < deadline - 5.0:
            riesz_deadline = time.time() + min(6.0, max(0.5, deadline - time.time() - 5.0))
            pts2, cr2 = _lbfgs_riesz(
                best,
                start_time=start_time,
                deadline=riesz_deadline,
                s=12.0,
                maxiter=150,
                ftol=1e-12,
                gtol=1e-10,
                maxfun=300,
            )
            if cr2 < best_cr:
                best_cr = cr2
                best = pts2.copy()

    return best, best_cr


# ---------------- initializers ----------------

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


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


# ---------------- main ----------------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s

    # Diverse initial configurations
    inits = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
        _icosahedron_based_points(N),
    ]

    best = _normalize(inits[0])
    best_cr = _covering_radius(best)

    # Phase 1: short Lloyd on each init to get into a reasonable basin
    for i, init in enumerate(inits):
        if deadline is not None and time.time() >= deadline - 20.0:
            break
        # allocate up to 4s per Lloyd pass
        lloyd_deadline = time.time() + min(4.0, max(0.5, deadline - time.time() - 20.0))
        pts, cr = _lloyd_iteration(
            _normalize(init),
            max_iters=150,
            tol=1e-10,
            start_time=start_time,
            deadline=lloyd_deadline,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Phase 2: short SA from best to escape local minima
    if deadline is not None and time.time() < deadline - 25.0:
        sa_deadline = time.time() + min(10.0, max(1.0, deadline - time.time() - 25.0))
        pts, cr = _simulated_annealing(
            best,
            start_time=start_time,
            deadline=sa_deadline,
            seed=123,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Phase 3: multi-beta LSE cascade (main refinement)
    if deadline is not None and time.time() < deadline - 10.0:
        pts, cr = _multi_beta_lse(
            best,
            start_time=start_time,
            deadline=deadline - 5.0,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Phase 4: final high-precision polish (tight LSE)
    if deadline is not None and time.time() < deadline - 2.0:
        polish_deadline = deadline - 1.0
        pts, cr = _lbfgs_smooth(
            best,
            start_time=start_time,
            deadline=polish_deadline,
            beta=1000.0,
            maxiter=400,
            ftol=1e-16,
            gtol=1e-15,
            maxfun=600,
        )
        if cr < best_cr:
            best_cr = cr
            best = pts.copy()

    # Final normalization and sanity check
    best = _normalize(best)
    # Ensure no NaN/Inf
    if not np.all(np.isfinite(best)):
        best = _fibonacci_points(N)

    return best
# EVOLVE-BLOCK-END
