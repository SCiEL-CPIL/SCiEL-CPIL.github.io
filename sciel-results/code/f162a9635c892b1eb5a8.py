"""
Sphere covering with 50 points: facet-shrink + L-BFGS soft-max gradient polish.
"""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import math

N = 50
_TIME_LIMIT = 140.0
_DEADLINE = time.time() + _TIME_LIMIT


def _time_remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius in degrees via convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_cr = 0.0
        for simplex in hull.simplices:
            a, b, c = pts[simplex]
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            if np.dot(n, a) < 0:
                n = -n

            v = np.cross(a - b, b - c)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                # Degenerate: use max edge midpoint distance
                d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                cr = max(d_ab, d_bc, d_ca) / 2.0
                max_cr = max(max_cr, cr)
                continue

            u = v / v_norm
            if np.dot(u, n) > 0:
                cc = u
            else:
                cc = -u

            cos_rad = np.clip(np.dot(cc, a), -1.0, 1.0)
            rad = np.arccos(cos_rad)
            max_cr = max(max_cr, rad)

        return np.degrees(max_cr)
    except Exception:
        return 180.0


def _facet_shrink_optimizer(pts, max_iter=300, initial_step=0.4, decay=0.995,
                            beta_start=2.0, beta_end=50.0):
    """Move points toward weighted circumcenters of incident facets."""
    pts = _normalize(pts.copy())
    n = len(pts)
    step_size = initial_step

    for it in range(max_iter):
        if _time_remaining() < 5.0:
            break

        try:
            hull = ConvexHull(pts)
        except Exception:
            break

        facet_data = []
        max_cr_global = 0.0

        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]

            v = np.cross(a - b, b - c)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            u = v / v_norm

            n_face = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n_face)
            if n_norm < 1e-12:
                continue
            n_face = n_face / n_norm
            if np.dot(n_face, a) < 0:
                n_face = -n_face

            if np.dot(u, n_face) > 0:
                cc = u
            else:
                cc = -u

            cos_rad = np.clip(np.dot(cc, a), -1.0, 1.0)
            rad = np.arccos(cos_rad)
            max_cr_global = max(max_cr_global, rad)
            facet_data.append((simplex, cc, rad))

        if not facet_data or max_cr_global < 1e-8:
            break

        progress = min(1.0, it / max_iter)
        beta = beta_start + (beta_end - beta_start) * progress

        weights_sum = np.zeros(n)
        vec_sum = np.zeros((n, 3))

        for simplex, cc, rad in facet_data:
            w = math.exp(beta * (rad - max_cr_global))
            for idx in simplex:
                weights_sum[idx] += w
                vec_sum[idx] += w * cc

        moved = False
        for i in range(n):
            if weights_sum[i] > 1e-10:
                target = vec_sum[i] / weights_sum[i]
                norm_t = np.linalg.norm(target)
                if norm_t > 1e-10:
                    target = target / norm_t
                    vec = target - pts[i]
                    vec_tan = vec - np.dot(vec, pts[i]) * pts[i]
                    norm_vec = np.linalg.norm(vec_tan)
                    if norm_vec > 1e-10:
                        pts[i] = pts[i] + step_size * vec_tan
                        moved = True

        if not moved:
            break

        pts = _normalize(pts)
        step_size *= decay
        if step_size < 1e-8:
            break

    return pts


def _soft_max_objective_and_grad(pts_flat, beta=80.0):
    """
    Soft-max covering radius (radians) and its gradient w.r.t. flat coords.
    Uses convex hull facet circumradii. Returns (value, grad_flat).
    """
    pts = _normalize(pts_flat.reshape(N, 3))
    n = len(pts)
    grad = np.zeros_like(pts)

    try:
        hull = ConvexHull(pts)
    except Exception:
        return 1.57, grad.ravel()

    crs = np.zeros(len(hull.simplices))
    ccs = []
    simplexes = []

    for k, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex]
        n_face = np.cross(b - a, c - a)
        nn = np.linalg.norm(n_face)
        if nn < 1e-15:
            crs[k] = 0.0
            ccs.append(np.zeros(3))
            simplexes.append(simplex)
            continue
        n_face = n_face / nn
        if np.dot(n_face, a) < 0:
            n_face = -n_face

        v = np.cross(a - b, b - c)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            # Degenerate: approximate with max edge midpoint
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            cr = max(d_ab, d_bc, d_ca) / 2.0
            crs[k] = cr
            # Gradient: push vertices apart from midpoint
            mid = (a + b) / 2.0
            mid = _normalize(mid)
            for idx in simplex:
                p = pts[idx]
                # d(cr)/d(p_i) ~ -dot(mid, p_i) * p_i_tangent component
                g = -np.dot(mid, p) * p
                g_tan = g - np.dot(g, p) * p
                grad[idx] += g_tan
            ccs.append(mid)
            simplexes.append(simplex)
            continue

        u = v / vn
        if np.dot(u, n_face) > 0:
            cc = u
        else:
            cc = -u

        cos_rad = np.clip(np.dot(cc, a), -1.0, 1.0)
        cr = np.arccos(cos_rad)
        crs[k] = cr
        ccs.append(cc)
        simplexes.append(simplex)

    # Soft-max
    crs_max = np.max(crs)
    if crs_max < 1e-12:
        return 0.0, grad.ravel()

    # Shift for numerical stability
    shifted = crs - crs_max
    weights = np.exp(beta * shifted)
    w_sum = np.sum(weights)
    if w_sum < 1e-30:
        weights = np.ones_like(crs) / len(crs)
        w_sum = 1.0
    else:
        weights = weights / w_sum

    soft_cr = crs_max + np.log(w_sum) / beta

    # Gradient: d(soft_cr)/d(pts_i) = sum_k weights[k] * d(cr_k)/d(pts_i)
    # d(cr_k)/d(pts_i) for circumcenter cc_k:
    # cr_k = arccos(dot(cc_k, a_k)) where a_k is a vertex
    # d(cr_k)/d(a) = -cc_k / sqrt(1 - dot(cc,a)^2)  (in 3D)
    # But we need the tangent component on the sphere.
    # For point p on sphere, d(arccos(dot(cc, p)))/d(p) = -cc / sin(cr)
    # The tangent gradient is the projection of this onto the tangent plane.
    for k in range(len(crs)):
        w = weights[k]
        if w < 1e-12:
            continue
        cc = ccs[k]
        cr = crs[k]
        if cr < 1e-10:
            continue
        sin_cr = np.sin(cr)
        if sin_cr < 1e-10:
            continue
        simplex = simplexes[k]
        for idx in simplex:
            p = pts[idx]
            # Gradient of arccos(dot(cc, p)) w.r.t. p (in R3)
            g3 = -cc / sin_cr
            # Project to tangent plane at p
            g_tan = g3 - np.dot(g3, p) * p
            grad[idx] += w * g_tan

    return soft_cr, grad.ravel()


def _l_bfgs_polish(pts, beta=80.0, max_iter=200):
    """Polish using L-BFGS on the soft-max covering radius."""
    pts = _normalize(pts.copy())
    x0 = pts.ravel()

    def obj_and_grad(x):
        val, g = _soft_max_objective_and_grad(x, beta=beta)
        return val, g

    try:
        result = minimize(
            obj_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={
                'maxiter': max_iter,
                'gtol': 1e-8,
                'ftol': 1e-12,
                'maxfun': 500,
            }
        )
        if result.success or result.fun < _soft_max_objective_and_grad(x0, beta=beta)[0]:
            new_pts = _normalize(result.x.reshape(N, 3))
            return new_pts
    except Exception:
        pass

    return pts


def _fibonacci_points():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(seed=None):
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _normalize(pts)


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT

    np.random.seed(42)

    best_pts = None
    best_cr = float('inf')

    # Phase 1: Multiple starts with Facet Shrink
    starts = [
        _fibonacci_points(),
        _random_points(seed=1),
        _random_points(seed=2),
        _random_points(seed=3),
        _random_points(seed=4),
        _random_points(seed=5),
    ]

    for s_idx, start in enumerate(starts):
        if _time_remaining() < 20.0:
            break

        pts = _facet_shrink_optimizer(
            start, max_iter=250, initial_step=0.35,
            decay=0.995, beta_start=2.0, beta_end=40.0
        )
        cr = _covering_radius(pts)

        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()

    # Phase 2: L-BFGS polish with increasing beta
    if best_pts is not None and _time_remaining() > 15.0:
        current = best_pts.copy()
        for beta in [50.0, 80.0, 120.0]:
            if _time_remaining() < 10.0:
                break
            polished = _l_bfgs_polish(current, beta=beta, max_iter=150)
            cr = _covering_radius(polished)
            if cr < best_cr:
                best_cr = cr
                best_pts = polished.copy()
                current = polished.copy()
            else:
                # If no improvement, break early
                break

    # Phase 3: Final facet-shrink micro-adjustment
    if best_pts is not None and _time_remaining() > 8.0:
        final = _facet_shrink_optimizer(
            best_pts, max_iter=50, initial_step=0.05,
            decay=0.99, beta_start=50.0, beta_end=100.0
        )
        cr = _covering_radius(final)
        if cr < best_cr:
            best_cr = cr
            best_pts = final.copy()

    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_points()

    best_pts = _normalize(best_pts)

    # Validate: no coincident points, not in closed hemisphere
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10):
        best_pts = _fibonacci_points()

    try:
        hull = ConvexHull(best_pts)
        inside = True
        for simplex in hull.simplices:
            tri = best_pts[simplex]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-15:
                n = n / n_norm
                if np.dot(n, tri[0]) < -1e-10:
                    inside = False
                    break
        if not inside:
            best_pts = _fibonacci_points()
    except Exception:
        best_pts = _fibonacci_points()

    return best_pts
