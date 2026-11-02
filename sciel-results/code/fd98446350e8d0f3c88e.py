"""Differentiable soft-Voronoi optimization for covering the sphere with 130 caps."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130
_T0 = time.time()
_TIME_LIMIT = 112.0  # safety margin before the 120s kill


def _grid_points():
    """High-resolution spherical grid (cubic-pyramid + icosahedral subdivision)."""
    pts = []
    # Cubic-pyramid projection grid (fine)
    m = 40
    for a in range(-m, m + 1):
        for b in range(-m, m + 1):
            pts.append((a, b, m))
            pts.append((a, -b, -m))
    # Icosahedral subdivision (adds points near vertices/edges)
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    ico /= np.linalg.norm(ico, axis=1, keepdims=True)
    # subdivide each triangle
    subdiv = []
    for i in range(len(ico)):
        for j in range(len(ico)):
            for k in range(len(ico)):
                if i < j < k:
                    # check if it's a face (roughly: use all combos, dedup later)
                    pass
    # simpler: add midpoints of all pairs that are ~1.057 apart (icosahedron edges)
    edge_len = 2.0 / np.sqrt(5.0 + 2.0 * phi)  # ~0.7236 chord -> angle ~41.8
    # just add midpoints of close pairs
    for i in range(len(ico)):
        for j in range(i + 1, len(ico)):
            d = np.linalg.norm(ico[i] - ico[j])
            if d < 0.8:  # edge
                mid = (ico[i] + ico[j]) / 2.0
                mid /= np.linalg.norm(mid)
                pts.append(mid)
    pts = np.array(pts, dtype=float)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    # deduplicate
    if len(pts) > 0:
        order = np.lexsort(pts.T)
        pts_sorted = pts[order]
        d = np.sqrt(np.sum(np.diff(pts_sorted, axis=0) ** 2, axis=1))
        mask = np.concatenate([[True], d > 1e-9])
        pts = pts_sorted[mask]
    return pts


_GRID = _grid_points()
_GRID_N = len(_GRID)


def _project_to_sphere(X):
    X = np.asarray(X, dtype=float).reshape(N, 3)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _soft_covering_radius(X, beta=300.0):
    """Smooth surrogate: log-sum-exp of squared geodesic distances (in deg^2).
    X: (N,3) on sphere; returns scalar approximating max angular distance in degrees.
    """
    X = _project_to_sphere(X)
    # squared chord distances / 4 -> approx sin^2(theta/2); use angle via arccos for accuracy
    # dot products
    D = _GRID @ X.T  # (G, N)
    D = np.clip(D, -1.0, 1.0)
    # squared angular distance in radians^2
    ang2 = np.arccos(D) ** 2
    # log-sum-exp over N for each grid point, then max over grid
    # For numerical stability:
    amax = np.max(ang2, axis=1)  # (G,)
    # logsumexp = amax + log(sum(exp(ang2 - amax)))
    s = np.log(np.sum(np.exp(ang2 - amax[:, None]), axis=1))
    lse = amax + s
    # soft max over grid: log(sum(exp(beta * lse))) / beta
    b = beta
    m = np.max(lse)
    softmax = m + np.log(np.sum(np.exp(b * (lse - m)))) / b
    return np.sqrt(softmax) * (180.0 / np.pi)  # degrees


def _objective(X):
    return _soft_covering_radius(X)


def _gradient(X):
    # numerical gradient is too slow; use L-BFGS-B with analytic-ish via finite diff? 
    # Instead, use L-BFGS-B with approx_grad? That's slow. Use analytic gradient of log-sum-exp.
    # We'll provide a gradient for speed.
    X = _project_to_sphere(X)
    D = _GRID @ X.T
    D = np.clip(D, -1.0, 1.0)
    ang = np.arccos(D)
    ang2 = ang ** 2
    amax = np.max(ang2, axis=1)
    w = np.exp(ang2 - amax[:, None])  # (G, N)
    w = w / np.sum(w, axis=1, keepdims=True)
    # d ang2 / d D = 2 * ang * (-1 / sqrt(1 - D^2))
    with np.errstate(divide='ignore', invalid='ignore'):
        dadD = -2.0 * ang / np.sqrt(np.maximum(1.0 - D * D, 1e-12))
    # d D / d X_j = _GRID  (D_gj = grid_g . X_j)
    # grad_j = sum_g w_gj * dadD_gj * grid_g
    grad = (w * dadD) @ _GRID  # (N, 3)
    # chain rule through projection: X = x / ||x||; dX/dx = (I - X X^T)/||x|| ... 
    # We pass X already on sphere; treat optimization in ambient space with projection in objective.
    # For L-BFGS-B we optimize the raw coordinates; gradient w.r.t. raw x:
    # dX/dx_k = (I - X X^T) / ||x||  (since X = x/||x||)
    # We'll compute using the current raw x.
    return grad


def _covering_radius_exact(X):
    """Exact covering radius via convex hull facet circumradii (degrees)."""
    X = _project_to_sphere(X)
    # remove duplicates
    X2 = []
    for p in X:
        if not any(np.linalg.norm(p - q) < 1e-8 for q in X2):
            X2.append(p)
    X2 = np.array(X2)
    if len(X2) < 4:
        return 180.0
    try:
        hull = ConvexHull(X2)
    except Exception:
        # fallback: use grid
        D = _GRID @ X2.T
        D = np.clip(D, -1.0, 1.0)
        ang = np.arccos(D)
        return np.max(ang) * (180.0 / np.pi)
    max_r = 0.0
    for face in hull.simplices:
        P = X2[face]
        # circumradius of triangle on sphere = circumradius of the planar triangle (chord)
        # For a triangle with vertices a,b,c, the circumradius (chord) R = abc/(4*Area)
        a = P[0]; b = P[1]; c = P[2]
        ab = b - a; ac = c - a; bc = c - b
        la = np.linalg.norm(ab); lb = np.linalg.norm(ac); lc = np.linalg.norm(bc)
        if la < 1e-12 or lb < 1e-12 or lc < 1e-12:
            continue
        # area
        area = 0.5 * np.linalg.norm(np.cross(ab, ac))
        if area < 1e-12:
            continue
        R = (la * lb * lc) / (4.0 * area)
        # angular circumradius = 2 * arcsin(R/2)
        if R > 2.0:
            R = 2.0
        ang_r = 2.0 * np.arcsin(R / 2.0)
        if ang_r > max_r:
            max_r = ang_r
    return max_r * (180.0 / np.pi)


def construct():
    global _T0
    deadline = _T0 + _TIME_LIMIT

    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    best_X = X0.copy()
    best_cr = _covering_radius_exact(X0)

    # Try a few random restarts + L-BFGS-B on the soft objective
    rng = np.random.default_rng(12345)
    n_restarts = 6
    for it in range(n_restarts):
        if time.time() > deadline - 20.0:
            break
        if it == 0:
            Xs = X0.copy()
        else:
            # random start
            Xs = rng.normal(size=(N, 3))
            Xs /= np.linalg.norm(Xs, axis=1, keepdims=True)
        try:
            res = minimize(_objective, Xs.reshape(-1), method='L-BFGS-B',
                           jac=None, options={'maxiter': 200, 'gtol': 1e-5,
                                              'ftol': 1e-7, 'maxfun': 4000})
            X_opt = _project_to_sphere(res.x.reshape(N, 3))
        except Exception:
            continue
        cr = _covering_radius_exact(X_opt)
        if cr < best_cr:
            best_cr = cr
            best_X = X_opt.copy()

    # Final local polish: gradient descent on exact-ish via small perturbations of worst facets
    # Use a simple iterative refinement: for each point, nudge toward the centroid of the
    # worst grid points it covers.
    for polish in range(3):
        if time.time() > deadline - 5.0:
            break
        X = best_X.copy()
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            # assign each grid point to nearest
            D = _GRID @ X.T
            D = np.clip(D, -1.0, 1.0)
            nearest = np.argmin(np.arccos(D), axis=1)
            # for each point, move toward mean of its grid points (in tangent space)
            for j in range(N):
                mask = nearest == j
                if not np.any(mask):
                    continue
                gpts = _GRID[mask]
                target = gpts.mean(axis=0)
                target /= np.linalg.norm(target)
                # small step
                X[j] = X[j] + 0.05 * (target - X[j])
                X[j] /= np.linalg.norm(X[j])
        cr = _covering_radius_exact(X)
        if cr < best_cr:
            best_cr = cr
            best_X = X.copy()

    return _project_to_sphere(best_X)
