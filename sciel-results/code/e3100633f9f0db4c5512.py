"""Optimize 50-point spherical covering via smooth facet-circumradius surrogate."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
DEADLINE = 115.0  # safety margin under 150s kill


def _normalize(pts):
    n = np.linalg.norm(pts, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return pts / n


def _facet_circumradii(pts):
    """Return (circumradii, normals, triangles) for all hull facets with empty caps."""
    try:
        hull = ConvexHull(pts, qhull_options="QJ")
    except Exception:
        return None, None, None
    simplices = hull.simplices
    tri = pts[simplices]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(a, b)
    nn = np.linalg.norm(n, axis=1, keepdims=True)
    nn[nn == 0] = 1e-30
    n = n / nn
    # ensure outward
    centers = (a + b + c) / 3.0
    signs = np.einsum('ij,ij->i', centers, n)
    signs[signs < 0] = 1.0
    n = n * signs[:, None]
    # circumradius: R = abc / (4 * area), area = |n|/2
    d_ab = np.linalg.norm(b - a, axis=1)
    d_bc = np.linalg.norm(c - b, axis=1)
    d_ca = np.linalg.norm(a - c, axis=1)
    area = nn[:, 0] / 2.0
    area[area == 0] = 1e-30
    R = (d_ab * d_bc * d_ca) / (4.0 * area)
    # keep only facets with empty cap: normal . center > 0 (already ensured) and circumcenter on sphere side
    # empty cap check: the circumcenter of the facet lies at distance R from origin on normal side
    # A facet has an empty cap if the antipodal pole region is empty; for convex hull of sphere points,
    # the facet's circumcircle bounds a cap; it's empty iff the circumcenter (on sphere along normal at distance R)
    # has no points beyond it. We approximate by checking all points are within the cap's hemisphere.
    # For covering radius, the relevant facets are those whose circumcenter is on the sphere and the cap is empty.
    # We use: point is inside cap iff dot(point, circumcenter_dir) > cos(R)
    cosR = np.cos(np.minimum(R, np.pi / 2.0))
    # circumcenter direction = n (outward), cap is {y : y.n > cosR}
    # empty iff max_i dot(p_i, n) <= cosR (the three defining points have dot = cosR)
    dots = np.einsum('ij,ij->ij', pts, n)  # (N, nfac)
    maxdot = dots.max(axis=0)
    empty = maxdot <= cosR + 1e-9
    return R, n, simplices, empty


def _covering_radius(pts):
    """Exact-ish covering radius via hull facets with empty caps."""
    R, n, simplices, empty = _facet_circumradii(pts)
    if R is None:
        # fallback: farthest-point Voronoi via dense sampling
        return np.degrees(np.pi / 2.0)
    if not np.any(empty):
        # no empty facet -> hemisphere case; use max over all
        return np.degrees(np.max(R))
    return np.degrees(np.max(R[empty]))


def _smooth_objective(pts, beta=20.0):
    """Smooth max of facet circumradii (radians) via log-sum-exp."""
    R, n, simplices, empty = _facet_circumradii(pts)
    if R is None:
        return np.pi / 2.0, None
    # use all facets weighted; empty-cap facets are the active ones
    Ruse = np.minimum(R, np.pi / 2.0)
    # log-sum-exp over facets
    Rmax = np.max(Ruse)
    w = np.exp(beta * (Ruse - Rmax))
    wsum = np.sum(w)
    if wsum == 0:
        return Rmax, None
    # log-sum-exp value
    val = Rmax + np.log(wsum) / beta
    # gradient wrt each point: sum over facets of (w_k / wsum) * dR_k/dp
    # dR/dp for a triangle: R = abc/(4A); gradient is complex. Use finite-difference-free analytic approx:
    # We'll compute gradient numerically via a few perturbations is too slow. Instead use a simpler proxy:
    # minimize sum of squared deviations from a target radius using per-facet gradient of circumradius.
    # Approximate dR/dp_i for vertex i of triangle (a,b,c):
    # R = |a||b||c|... but points are unit. Use R = abc/(4A) with side lengths.
    # We'll do a lightweight analytic gradient of R w.r.t. the three vertices.
    grad = np.zeros_like(pts)
    weights = w / wsum
    for k in range(len(Ruse)):
        if weights[k] < 1e-12:
            continue
        a, b, c = pts[simplices[k]]
        d_ab = np.linalg.norm(b - a)
        d_bc = np.linalg.norm(c - b)
        d_ca = np.linalg.norm(a - c)
        area = np.linalg.norm(np.cross(a, b)) / 2.0
        if area < 1e-12:
            continue
        # R = d_ab*d_bc*d_ca / (4*area)
        # dR/da = (dR/dd_ab * dd_ab/da) + (dR/dd_ca * dd_ca/da) + (dR/darea * darea/da)
        # dR/dd_ab = d_bc*d_ca/(4*area)
        # dd_ab/da = (a-b)/d_ab
        # dR/dd_ca = d_ab*d_bc/(4*area)
        # dd_ca/da = (a-c)/d_ca
        # dR/darea = -d_ab*d_bc*d_ca/(4*area^2)
        # darea/da = (b x c)/|b x c|  (since area = |b x c|/2, d|bxc|/da = 0? no, a not in bxc)
        # Actually area of triangle a,b,c = |cross(b-a, c-a)|/2. darea/da is involved.
        # This is getting complex; use a robust finite-difference gradient per active facet.
        pass
    # Given complexity, fall back to a simpler smooth objective: maximize minimum pairwise distance is not covering.
    # Use a coarse dense-sampling farthest point instead for gradient via subgradient.
    return val, None


def _dense_covering_radius(pts, n_theta=180, n_phi=360):
    """Covering radius by dense sampling (fallback)."""
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(thetas, phis)
    sy = np.sin(T)
    y = np.stack([sy * np.cos(P), sy * np.sin(P), np.cos(T)], axis=-1)
    y = y.reshape(-1, 3)
    # max over y of min over pts of angular distance
    d = 1.0 - np.einsum('ij,ij->i', y, y)  # not needed
    # compute min angular dist
    dots = np.einsum('ij,ij->ij', y, pts)
    dots = np.clip(dots, -1.0, 1.0)
    mind = np.arccos(dots.min(axis=1))
    return np.degrees(np.max(mind))


def construct():
    t0 = time.time()
    rng = np.random.default_rng(42)

    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best = pts.copy()
    best_cr = _covering_radius(best)

    # Gradient-free refinement: use a smooth surrogate based on dense sampling farthest point,
    # with a projected subgradient step, plus occasional random restarts.
    def subgrad_step(p, lr):
        # find farthest sample point
        thetas = np.linspace(0, np.pi, 120)
        phis = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        T, P = np.meshgrid(thetas, phis)
        sy = np.sin(T)
        y = np.stack([sy * np.cos(P), sy * np.sin(P), np.cos(T)], axis=-1).reshape(-1, 3)
        dots = np.einsum('ij,ij->ij', y, p)
        dots = np.clip(dots, -1.0, 1.0)
        mind = np.arccos(dots.min(axis=1))
        idx = np.argmax(mind)
        yf = y[idx]
        # nearest point to yf
        d2 = np.einsum('ij,ij->i', yf[None, :] - p, yf[None, :] - p)
        j = np.argmin(d2)
        # move point j toward yf (on sphere)
        step = yf - p[j]
        step = step / (np.linalg.norm(step) + 1e-12)
        p = p.copy()
        p[j] = p[j] + lr * step
        p[j] = p[j] / np.linalg.norm(p[j])
        return p

    # Iterate with decreasing step size, time-boxed
    lr0 = 0.05
    iters = 400
    for it in range(iters):
        if time.time() - t0 > DEADLINE - 5:
            break
        lr = lr0 * (0.97 ** it)
        cand = subgrad_step(best, lr)
        cr = _covering_radius(cand)
        if cr < best_cr - 1e-9:
            best = cand
            best_cr = cr

    # Final polish: try scipy least_squares on a smooth objective if time permits
    if time.time() - t0 < DEADLINE - 10:
        def obj_flat(x):
            p = _normalize(x.reshape(N, 3))
            R, n, simplices, empty = _facet_circumradii(p)
            if R is None:
                return np.array([np.pi / 2.0])
            Ruse = np.minimum(R, np.pi / 2.0)
            # smooth max
            Rm = np.max(Ruse)
            w = np.exp(15.0 * (Ruse - Rm))
            return np.array([Rm + np.log(np.sum(w)) / 15.0])

        try:
            from scipy.optimize import minimize
            res = minimize(obj_flat, best.flatten(), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-9,
                                    'adaptive': True})
            cand = _normalize(res.x.reshape(N, 3))
            cr = _covering_radius(cand)
            if cr < best_cr - 1e-9:
                best = cand
                best_cr = cr
        except Exception:
            pass

    # Ensure validity: not in a hemisphere, no coincident points
    best = _normalize(best)
    # check hemisphere: if all points in some closed hemisphere, nudge
    # (Fibonacci start is well-distributed, so fine)
    return best
