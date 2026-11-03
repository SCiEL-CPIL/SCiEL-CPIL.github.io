"""Packing 50 equal circles in a unit circle: ring-structure global search + soft-min
surrogate polish + barrier-based radius maximization (epigraph on r)."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

# --- Ring structure: 1 centre + rings of 6,12,18,13 = 50 ---
RING_COUNTS = [6, 12, 18, 13]
N_RINGS = len(RING_COUNTS)

# Precompute upper-triangular pair indices (pair (i,j), i<j)
_IU_I, _IU_J = np.triu_indices(N, k=1)
_NPAIR = _IU_I.shape[0]


def _compute_score(points):
    """Exact packing score: min(min_pair_dist/2, min_boundary_dist)."""
    if points is None or points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    radii = np.hypot(points[:, 0], points[:, 1])
    if np.any(radii > 1.0 + 1e-12):
        return 0.0
    min_boundary = np.min(1.0 - radii)
    if min_boundary < 0.0:
        return 0.0
    d = cdist(points, points)
    min_pair = d[_IU_I, _IU_J].min() / 2.0
    return min(min_pair, min_boundary)


def _clamp(points):
    """Clamp points to the closed unit disk (return a new array)."""
    p = points.copy()
    r = np.hypot(p[:, 0], p[:, 1])
    mask = r > 1.0
    if np.any(mask):
        p[mask] /= r[mask, np.newaxis]
    return p


def _soft_min_surrogate(points, beta=100.0):
    """Smooth surrogate for the packing radius (soft-min of all constraints)."""
    pts = points
    d = cdist(pts, pts)
    pair_vals = d[_IU_I, _IU_J] / 2.0
    radii = np.hypot(pts[:, 0], pts[:, 1])
    boundary_vals = 1.0 - radii
    all_vals = np.concatenate([pair_vals, boundary_vals])
    m = all_vals.min()
    e = np.exp(-beta * (all_vals - m))
    return m - np.log(e.sum()) / beta


def _soft_min_obj(x, beta=100.0):
    """Objective = -soft_min + hard boundary penalty (for L-BFGS-B)."""
    pts = x.reshape(N, 2)
    r = _soft_min_surrogate(pts, beta)
    norms = np.linalg.norm(pts, axis=1)
    pen = 1000.0 * np.sum(np.maximum(0.0, norms - 1.0) ** 2)
    return -r + pen


def _decode_ring(params):
    """Decode ring parameters into N points. params = [r_0..r_{K-1}, off_0..off_{K-1}]."""
    points = [np.array([0.0, 0.0])]
    for k, nk in enumerate(RING_COUNTS):
        rk = params[k]
        off = params[N_RINGS + k]
        for m in range(nk):
            ang = 2.0 * np.pi * m / nk + off
            points.append(np.array([rk * np.cos(ang), rk * np.sin(ang)]))
    return np.array(points[:N])


def _ring_global_search(deadline):
    """Coarse grid search over ring radii/offsets; keep the best configuration."""
    best_pts = None
    best_score = -1.0
    # Candidate radii per ring (inner to outer)
    r_choices = {
        0: [0.28, 0.30, 0.32, 0.34, 0.36, 0.38],
        1: [0.55, 0.58, 0.61, 0.64, 0.67, 0.70],
        2: [0.78, 0.81, 0.84, 0.87, 0.90],
        3: [0.92, 0.94, 0.96, 0.98],
    }
    off_choices = [0.0, np.pi / 6, np.pi / 12, np.pi / 24]
    total = 1
    for k in range(N_RINGS):
        total *= len(r_choices[k]) * len(off_choices)
    count = 0
    for r0 in r_choices[0]:
        for r1 in r_choices[1]:
            for r2 in r_choices[2]:
                for r3 in r_choices[3]:
                    for o0 in off_choices:
                        for o1 in off_choices:
                            for o2 in off_choices:
                                for o3 in off_choices:
                                    if time.time() > deadline - 20:
                                        return best_pts
                                    params = [r0, r1, r2, r3, o0, o1, o2, o3]
                                    pts = _decode_ring(params)
                                    s = _compute_score(pts)
                                    if s > best_score:
                                        best_score = s
                                        best_pts = pts.copy()
                                    count += 1
    return best_pts


def _polish_smooth(points, deadline):
    """Progressive soft-min polish with increasing beta (smooth surrogate)."""
    current = _clamp(points)
    current_score = _compute_score(current)
    for beta in [20, 50, 100, 200, 400, 800]:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(
                _soft_min_obj,
                current.flatten(),
                args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-14, 'maxls': 60},
            )
            cand = _clamp(res.x.reshape(N, 2))
            if np.all(np.isfinite(cand)):
                s = _compute_score(cand)
                if s > current_score:
                    current = cand
                    current_score = s
        except Exception:
            pass
    return current, current_score


def _barrier_max_radius(points, deadline):
    """Maximize radius r with log-barrier on non-overlap and boundary constraints.

    Epigraph form: maximize r subject to
        |c_i - c_j| >= 2r   (i<j)
        |c_i| + r <= 1       (i)
    Barrier:  L = -r - mu * sum_i log(1 - |c_i| - r)
                          - mu * sum_{i<j} log(|c_i - c_j| - 2r)
    Minimize L over (x, r) with r in (0, 1).
    """
    current = _clamp(points)
    current_score = _compute_score(current)
    best = (current, current_score)
    mu = 0.05
    for _ in range(14):
        if time.time() > deadline - 4:
            break

        def barrier_obj(x):
            r = x[-1]
            pts = x[:2 * N].reshape(N, 2)
            r = min(max(r, 1e-4), 0.999)
            norms = np.linalg.norm(pts, axis=1)
            bnd = 1.0 - norms - r
            pd = cdist(pts, pts)
            pd = pd[_IU_I, _IU_J] - 2.0 * r
            # guard against log of non-positive
            bnd = np.maximum(bnd, 1e-9)
            pd = np.maximum(pd, 1e-9)
            return -r - mu * (np.sum(np.log(bnd)) + np.sum(np.log(pd)))

        def barrier_grad(x):
            r = x[-1]
            pts = x[:2 * N].reshape(N, 2)
            r = min(max(r, 1e-4), 0.999)
            g = np.zeros(2 * N + 1)
            norms = np.linalg.norm(pts, axis=1)
            bnd = np.maximum(1.0 - norms - r, 1e-9)
            # boundary terms
            for i in range(N):
                inv = 1.0 / bnd[i]
                # d(-mu log bnd_i)/d c_i = -mu * (-(pts_i/norm_i)) / bnd_i
                if norms[i] > 1e-12:
                    g[2 * i] += -mu * (-pts[i, 0] / norms[i]) * inv
                    g[2 * i + 1] += -mu * (-pts[i, 1] / norms[i]) * inv
                # d(-mu log bnd_i)/d r = -mu * (-1)/bnd_i
                g[-1] += -mu * (-1.0) * inv
            # pair terms
            pd = cdist(pts, pts)
            pd = pd[_IU_I, _IU_J]
            pdg = np.maximum(pd - 2.0 * r, 1e-9)
            for a in range(_NPAIR):
                i = _IU_I[a]; j = _IU_J[a]
                diff = pts[i] - pts[j]
                dist = pd[a]
                if dist < 1e-12:
                    continue
                inv = 1.0 / pdg[a]
                # d(-mu log(dist-2r))/d c_i = -mu * (diff/dist) / (dist-2r)
                g[2 * i] += -mu * (diff[0] / dist) * inv
                g[2 * i + 1] += -mu * (diff[1] / dist) * inv
                g[2 * j] += -mu * (-diff[0] / dist) * inv
                g[2 * j + 1] += -mu * (-diff[1] / dist) * inv
                # d(-mu log(dist-2r))/d r = -mu * (-2)/(dist-2r)
                g[-1] += -mu * (-2.0) * inv
            # objective -r gives d/d r = -1
            g[-1] += -1.0
            return g

        x0 = np.concatenate([current.flatten(), [current_score]])
        try:
            res = minimize(
                barrier_obj,
                x0,
                jac=barrier_grad,
                method='L-BFGS-B',
                bounds=[(0.0, 1.0)] * (2 * N) + [(1e-4, 0.999)],
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 40},
            )
            cand = _clamp(res.x[:2 * N].reshape(N, 2))
            if np.all(np.isfinite(cand)):
                s = _compute_score(cand)
                if s > best[1]:
                    best = (cand, s)
        except Exception:
            pass
        mu *= 0.7  # reduce barrier strength
    return best


def construct():
    t0 = time.time()
    deadline = t0 + 142.0  # safety margin before 150s kill

    best_pts = None
    best_score = 0.0

    def consider(pts):
        nonlocal best_pts, best_score
        if pts is None:
            return
        pts = _clamp(pts)
        if not np.all(np.isfinite(pts)):
            return
        s = _compute_score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 1: ring-structure global search (cheap, diverse starts)
    if time.time() < deadline - 60:
        ring_pts = _ring_global_search(deadline - 60)
        if ring_pts is not None:
            consider(ring_pts)

    # Phase 2: smooth soft-min polish from the best ring config
    if best_pts is not None and time.time() < deadline - 30:
        polished, s = _polish_smooth(best_pts, deadline - 30)
        consider(polished)

    # Phase 3: barrier-based radius maximization (epigraph on r)
    if best_pts is not None and time.time() < deadline - 10:
        bpts, bs = _barrier_max_radius(best_pts, deadline - 10)
        consider(bpts)

    # Phase 4: final high-precision smooth polish
    if best_pts is not None and time.time() < deadline - 3:
        polished, s = _polish_smooth(best_pts, deadline - 3)
        consider(polished)

    if best_pts is None:
        # Deterministic fallback ring
        params = [0.32, 0.62, 0.85, 0.96, 0.0, np.pi / 12, 0.0, np.pi / 13]
        best_pts = _clamp(_decode_ring(params))
        s = _compute_score(best_pts)
        if s > best_score:
            best_score = s

    # Final validation / safety
    best_pts = _clamp(best_pts)
    if not np.all(np.isfinite(best_pts)) or _compute_score(best_pts) <= 0.0:
        params = [0.30, 0.60, 0.84, 0.95, 0.0, np.pi / 6, 0.0, np.pi / 12]
        best_pts = _clamp(_decode_ring(params))
    return best_pts

# EVOLVE-BLOCK-END
