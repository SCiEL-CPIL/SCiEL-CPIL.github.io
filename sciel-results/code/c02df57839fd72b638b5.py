"""Rewritten optimizer for LJ-38 cluster global minimum.

Strategy: build a diverse set of high-quality initial configurations
(FCC truncated octahedron = known global min, icosahedral, random shells),
polish each with a robust L-BFGS local minimizer using the analytic LJ
gradient, and return the lowest-energy valid configuration found within
the wall-clock budget.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 150.0
START_TIME = time.time()


def _time_left():
    return max(0.0, DEADLINE - (time.time() - START_TIME))


def _lj_energy(pos):
    """Total LJ energy: 4*sum(r^-12 - r^-6) over all pairs."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    # pairwise displacement vectors (n, n, 3)
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)  # avoid self
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    e = 4.0 * (inv_r12 - inv_r6)
    return 0.5 * np.sum(e)  # each pair counted twice


def _lj_energy_grad(pos):
    """Energy and analytic gradient of total LJ energy."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]   # diff[i,j] = pos[i]-pos[j]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    # pair energy matrix (i,j) = 4*(r^-12 - r^-6)
    e = 4.0 * (inv_r12 - inv_r6)
    energy = 0.5 * np.sum(e)
    # gradient: dE/dpos[i] = sum_j 4*(-12 r^-14 + 6 r^-8) * (pos[i]-pos[j])
    # = sum_j ( -48 r^-14 + 24 r^-8 ) * diff[i,j]
    # factor = (-48 * inv_r12 + 24 * inv_r6) / r2  ... since r^-14 = r^-12 / r^2
    # Actually: d/dx [4(r^-12 - r^-6)] = 4(-12 r^-13 dr/dx + 6 r^-7 dr/dx)
    # dr/dx_i = (x_i - x_j)/r  =>  gradient wrt pos[i] = 4(-12 r^-14 + 6 r^-8)*(pos[i]-pos[j])
    grad_factor = 4.0 * (-12.0 * inv_r12 + 6.0 * inv_r6) * inv_r2  # = 4*(-12 r^-14 + 6 r^-8)
    # grad[i] = sum_j grad_factor[i,j] * diff[i,j]
    grad = np.einsum('ij,ijk->ik', grad_factor, diff)
    return energy, grad


def _minimize_lj(pos0, maxiter=400, timeout=60.0):
    """Polish a configuration with L-BFGS-B using analytic gradient."""
    def fun(x):
        e, g = _lj_energy_grad(x.reshape(-1, 3))
        return e, g
    res = minimize(fun, pos0.ravel(), jac=True, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12})
    return res.x.reshape(-1, 3), res.fun


def _minimize_lj_with_deadline(pos0, maxiter=300):
    """Polish but abort if we are getting close to the global deadline."""
    t0 = time.time()
    def fun(x):
        if time.time() - t0 > 90:  # per-call safety
            return 1e9, np.zeros(x.shape)
        e, g = _lj_energy_grad(x.reshape(-1, 3))
        return e, g
    try:
        res = minimize(fun, pos0.ravel(), jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-12})
        return res.x.reshape(-1, 3), res.fun
    except Exception:
        return pos0, _lj_energy(pos0)


def _check_min_distance(pos, min_d=0.5):
    """True if all pairwise distances >= min_d."""
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_d - 1e-9


def _fcc_truncated_octahedron():
    """Build an FCC truncated octahedron (known global min for LJ38).

    The LJ38 global minimum is a truncated octahedron derived from FCC.
    We generate FCC lattice points near the origin and keep the 38 closest
    to the centroid, then center the cluster.
    """
    # FCC primitive vectors
    a = 1.122462048309373  # 2^(1/6) * ... actually LJ min r = 2^(1/6) ≈ 1.12246
    # Use the standard: equilibrium distance r0 = 2**(1/6)
    r0 = 2.0 ** (1.0 / 6.0)
    # FCC lattice: points (i,j,k)*r0 with (i+j+k) even, in a cubic cell of side 2*r0? 
    # Simpler: generate FCC points as all (x,y,z) where x,y,z in {0, r0, 2r0, ...}
    # and (x+y+z)/r0 has integer coordinates with even sum.
    # Let's generate on integer grid scaled by r0 with FCC condition.
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    pts.append(np.array([i, j, k]) * r0)
    pts = np.array(pts)
    # center
    c = pts.mean(axis=0)
    pts = pts - c
    # sort by distance from center, take 38 closest
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)
    pts = pts[idx[:N]]
    # re-center
    pts = pts - pts.mean(axis=0)
    return pts


def _icosahedral_start():
    """Icosahedral shell configuration (a strong local-minimum competitor)."""
    r0 = 2.0 ** (1.0 / 6.0)
    # icosahedron vertices (12)
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [t, 0, -1], [-t, 0, 1], [-t, 0, -1],
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    # icosahedron has 12 vertices, 30 edges. For 38 atoms we need inner + outer shells.
    # Known icosahedral LJ38: 1 center + 12 (inner icosahedron) + 12 (outer icosahedron) + 13? 
    # Simpler: use 1 + 12 + 12 + 13 is awkward. Use 1 center, 12 at radius r1, 25 more.
    # Build: center + 12 icosahedron vertices at radius R1 + 12 at radius R2 (same directions, larger)
    # + remaining 13 placed on edges/face centers.
    # For a valid diverse start, place: 1 center, 12 at R1, 12 at R2, and 13 at edge midpoints scaled.
    R1 = r0
    R2 = r0 * 1.9  # outer shell
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v * R1)
    for v in verts:
        pts.append(v * R2)
    # need 13 more: use edge midpoints of the icosahedron at radius ~R1*1.4
    # icosahedron edges: pairs of vertices with shortest distance
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.linalg.norm(verts[i] - verts[j]) < 1.5:  # edge threshold in unit icosahedron
                edges.append((i, j))
    # take first 13 edge midpoints
    for (i, j) in edges[:13]:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid) * (r0 * 1.45)
        pts.append(mid)
    pts = np.array(pts[:N])
    return pts - pts.mean(axis=0)


def _random_cluster_start(seed):
    """Random cluster: atoms placed on concentric shells to avoid overlap."""
    rng = np.random.RandomState(seed)
    r0 = 2.0 ** (1.0 / 6.0)
    pts = [np.zeros(3)]
    # add atoms one by one on a sphere of increasing radius, keeping min distance
    for i in range(1, N):
        for attempt in range(2000):
            # random direction
            v = rng.randn(3)
            v = v / (np.linalg.norm(v) + 1e-12)
            # radius grows with index
            radius = r0 * (1.0 + 0.15 * i)
            cand = v * radius
            # check min distance to existing
            if len(pts) > 0:
                d = np.linalg.norm(np.array(pts) - cand, axis=1)
                if np.min(d) < r0 * 0.95:
                    continue
            pts.append(cand)
            break
        else:
            # fallback: place at a slightly larger radius
            cand = v * (r0 * (1.0 + 0.2 * i))
            pts.append(cand)
    pts = np.array(pts)
    return pts - pts.mean(axis=0)


def construct():
    best_pos = None
    best_e = np.inf

    # Candidate generators
    candidates = []
    try:
        candidates.append(_fcc_truncated_octahedron())
    except Exception:
        pass
    try:
        candidates.append(_icosahedral_start())
    except Exception:
        pass
    # several random cluster starts with different seeds
    for s in (1, 7, 42, 123, 999):
        try:
            candidates.append(_random_cluster_start(s))
        except Exception:
            pass

    # Also a simple-cubic seed (the original) as a fallback
    try:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        candidates.append(np.array(points[:N], dtype=float) * 1.12)
    except Exception:
        pass

    # Polish each candidate, keep best
    for pos in candidates:
        if _time_left() < 5.0:
            break
        try:
            polished, e = _minimize_lj_with_deadline(pos, maxiter=300)
            if _check_min_distance(polished, 0.5):
                if e < best_e:
                    best_e = e
                    best_pos = polished.copy()
        except Exception:
            # if polishing fails, use the raw candidate if valid
            if _check_min_distance(pos, 0.5):
                e = _lj_energy(pos)
                if e < best_e:
                    best_e = e
                    best_pos = pos.copy()

    # If nothing valid was found, return a guaranteed-valid simple cubic seed
    if best_pos is None:
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12

    # Final safety: ensure min distance
    if not _check_min_distance(best_pos, 0.5):
        # nudge apart by scaling
        c = best_pos.mean(axis=0)
        best_pos = (best_pos - c) * 1.2 + c
        if not _check_min_distance(best_pos, 0.5):
            # fall back to simple cubic
            side = int(np.ceil(N ** (1 / 3)))
            points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
            points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
            best_pos = np.array(points[:N], dtype=float) * 1.12

    return best_pos
