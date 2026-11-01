import numpy as np
import time
import random
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0

# Unit hexagon vertices (circumradius = side = 1), angles 0,60,...,300
UNIT_VERTS = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
# Outward edge normals (apothem directions), angles 30,90,...,330
OUTER_NORMALS = np.array([[np.cos(np.radians(60*i+30)), np.sin(np.radians(60*i+30))] for i in range(6)])
NORM = OUTER_NORMALS  # unit vectors
SQ3 = np.sqrt(3.0)
PEN_W = 20.0          # penalty weight for soft containment/overlap


def _rot_matrix(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _verts_from(centers):
    """centers (N,3) [x,y,deg] -> (N,6,2) vertices."""
    c = centers[:, 0:1]
    s = centers[:, 1:2]
    a = np.radians(centers[:, 2:3])
    cos = np.cos(a); sin = np.sin(a)
    R = np.stack([np.stack([cos, -sin], axis=1),
                  np.stack([sin,  cos], axis=1)], axis=1)  # (N,2,2)
    # rotate UNIT_VERTS by R
    uv = UNIT_VERTS[None, :, :]                       # (1,6,2)
    rv = np.einsum('nij,kjd->nik', R, uv)             # (N,6,2)
    return rv + np.stack([c, s], axis=1)[:, None, :]  # (N,6,2)


def _clearances(verts, outer_center, outer_angle):
    """Returns (min_pair_gap, min_outer_gap).
    Positive = separated / inside. Negative = overlap / outside.
    """
    # ---- pairwise SAT (batched) ----
    e = np.roll(verts, -1, axis=1) - verts            # (N,6,2)
    nrm = np.linalg.norm(e, axis=2)
    nrm[nrm < 1e-12] = 1.0
    normals = np.stack((-e[..., 1], e[..., 0]), axis=2) / nrm[..., None]  # (N,6,2)

    iu = np.triu_indices(N, 1)
    vi = verts[iu[0]]                                 # (M,6,2)
    vj = verts[iu[1]]
    ni = normals[iu[0]]
    nj = normals[iu[1]]
    axes = np.concatenate([ni, nj], axis=1)           # (M,12,2)
    pi = np.einsum('mij,mik->mkj', vi, axes)          # (M,12,6)
    pj = np.einsum('mij,mik->mkj', vj, axes)
    sep = np.maximum(pi.min(axis=2) - pj.max(axis=2),
                     pj.min(axis=2) - pi.max(axis=2)) # (M,12)
    min_pair = float(sep.min())

    # ---- outer containment (support / apothem) ----
    Ro = _rot_matrix(-outer_angle)
    vrel = verts - np.array(outer_center)[None, None, :]
    vrot = np.einsum('ij,mjk->mik', Ro, vrel)         # (N,6,2)
    projs = np.einsum('mij,jk->mik', vrot, NORM.T)    # (N,6,6)
    maxproj = np.abs(projs).max(axis=1)               # (N,6)
    apothem = 0.0                                     # side unknown -> use gap in units of apothem
    # express as clearance relative to a unit-apothem boundary, scaled later
    min_outer = float((0.5 * SQ3 * 1.0 - maxproj).min())  # placeholder, side handled separately
    return min_pair, maxproj.max()


def outer_side(centers, outer_center, outer_angle):
    """Minimal outer side that contains all hexagons (apothem-based)."""
    verts = _verts_from(centers)
    Ro = _rot_matrix(-outer_angle)
    vrel = verts - np.array(outer_center)[None, None, :]
    vrot = np.einsum('ij,mjk->mik', Ro, vrel)
    projs = np.einsum('mij,jk->mik', vrot, NORM.T)
    maxproj = np.abs(projs).max()
    return 2.0 * maxproj / SQ3


def validity_gap(centers, outer_center, outer_side, outer_angle):
    """Return min of (pair_gap, outer_gap). >= -tol means valid."""
    verts = _verts_from(centers)
    e = np.roll(verts, -1, axis=1) - verts
    nrm = np.linalg.norm(e, axis=2); nrm[nrm < 1e-12] = 1.0
    normals = np.stack((-e[..., 1], e[..., 0]), axis=2) / nrm[..., None]
    iu = np.triu_indices(N, 1)
    vi = verts[iu[0]]; vj = verts[iu[1]]
    ni = normals[iu[0]]; nj = normals[iu[1]]
    axes = np.concatenate([ni, nj], axis=1)
    pi = np.einsum('mij,mik->mkj', vi, axes)
    pj = np.einsum('mij,mik->mkj', vj, axes)
    sep = np.maximum(pi.min(axis=2) - pj.max(axis=2),
                     pj.min(axis=2) - pi.max(axis=2))
    min_pair = float(sep.min())

    Ro = _rot_matrix(-outer_angle)
    vrel = verts - np.array(outer_center)[None, None, :]
    vrot = np.einsum('ij,mjk->mik', Ro, vrel)
    projs = np.einsum('mij,jk->mik', vrot, NORM.T)
    maxproj = np.abs(projs).max()
    apothem = outer_side * SQ3 / 2.0
    min_outer = float(apothem - maxproj)
    return min(min_pair, min_outer)


def objective(centers, outer_center, outer_angle):
    side = outer_side(centers, outer_center, outer_angle)
    verts = _verts_from(centers)
    # pairwise
    e = np.roll(verts, -1, axis=1) - verts
    nrm = np.linalg.norm(e, axis=2); nrm[nrm < 1e-12] = 1.0
    normals = np.stack((-e[..., 1], e[..., 0]), axis=2) / nrm[..., None]
    iu = np.triu_indices(N, 1)
    vi = verts[iu[0]]; vj = verts[iu[1]]
    ni = normals[iu[0]]; nj = normals[iu[1]]
    axes = np.concatenate([ni, nj], axis=1)
    pi = np.einsum('mij,mik->mkj', vi, axes)
    pj = np.einsum('mij,mik->mkj', vj, axes)
    sep = np.maximum(pi.min(axis=2) - pj.max(axis=2),
                     pj.min(axis=2) - pi.max(axis=2))
    min_pair = float(sep.min())
    # outer
    Ro = _rot_matrix(-outer_angle)
    vrel = verts - np.array(outer_center)[None, None, :]
    vrot = np.einsum('ij,mjk->mik', Ro, vrel)
    projs = np.einsum('mij,jk->mik', vrot, NORM.T)
    maxproj = float(np.abs(projs).max())
    apothem = side * SQ3 / 2.0
    min_outer = apothem - maxproj
    pen = 0.0
    if min_pair < 0: pen += PEN_W * min_pair
    if min_outer < 0: pen += PEN_W * min_outer
    return side + pen


def polish(centers, outer_center, outer_angle, budget):
    x0 = np.array(centers, dtype=float).ravel()
    t0 = time.time()
    try:
        res = minimize(
            lambda x: objective(x.reshape(N, 3), outer_center, outer_angle),
            x0, method='L-BFGS-B',
            options={'maxiter': 60, 'maxfun': 400, 'ftol': 1e-12, 'gtol': 1e-8},
        )
        if time.time() - t0 > budget:
            return x0.reshape(N, 3)
        return res.x.reshape(N, 3)
    except Exception:
        return x0.reshape(N, 3)


def sa_run(centers, outer_center, outer_angle, budget):
    t0 = time.time()
    cur = np.array(centers, dtype=float)
    cur_cost = objective(cur, outer_center, outer_angle)
    best = cur.copy(); best_cost = cur_cost
    T = 0.35; T_min = 0.0015; alpha = 0.9965
    step_p = 0.18; step_a = 4.0
    while time.time() - t0 < budget and T > T_min:
        i = random.randrange(N)
        new = cur.copy()
        new[i, 0] += random.uniform(-step_p, step_p)
        new[i, 1] += random.uniform(-step_p, step_p)
        new[i, 2] += random.uniform(-step_a, step_a)
        nc = objective(new, outer_center, outer_angle)
        d = nc - cur_cost
        if d < 0 or random.random() < np.exp(-d / T):
            cur = new; cur_cost = nc
            if nc < best_cost:
                best_cost = nc; best = new.copy()
        T *= alpha
    return best, best_cost


def init_hex_lattice():
    # 12 unit hexagons on a honeycomb lattice (side spacing 2*apothem = sqrt(3) in a
    # flat-top arrangement). Use a 4-3-4-1 style cluster that is compact.
    dx = SQ3        # horizontal spacing between column centers (flat-top)
    dy = 1.5        # vertical offset per row
    cols = [(0, 0), (dx, 0), (2*dx, 0), (3*dx, 0),
            (dx/2, dy), (3*dx/2, dy),
            (dx, -dy), (2*dx, -dy),
            (0, 2*dy), (dx, 2*dy),
            (dx/2, -dy*2), (3*dx/2, -dy*2)]
    return np.array([[x, y, 0.0] for (x, y) in cols])


def init_circular(r):
    arr = []
    for i in range(N):
        th = 2*np.pi*i/N
        arr.append([r*np.cos(th), r*np.sin(th), np.degrees(th)])
    return np.array(arr)


def init_random(seed):
    rng = np.random.RandomState(seed)
    arr = rng.uniform(-3.5, 3.5, size=(N, 2))
    ang = rng.uniform(0, 60, size=N)
    return np.hstack([arr, ang[:, None]])


def construct():
    t_start = time.time()
    outer_center = (0.0, 0.0)
    outer_angle = 0.0

    starts = [
        init_hex_lattice(),
        init_circular(2.6),
        init_circular(2.2),
        init_random(1), init_random(2), init_random(3),
        init_random(4), init_random(5), init_random(6),
    ]

    best_centers = None
    best_side = float('inf')

    def remaining():
        return DEADLINE - (time.time() - t_start)

    for k, st in enumerate(starts):
        if remaining() < 8:
            break
        # SA budget: leave room for polish + validation
        sa_budget = min(12.0, max(3.0, remaining() - 10))
        try:
            sa_c, _ = sa_run(st, outer_center, outer_angle, sa_budget)
            if remaining() < 5:
                sa_c = st
            else:
                pol_c = polish(sa_c, outer_center, outer_angle, max(1.0, remaining() - 5))
                # keep the better of SA and polish
                s_sa = outer_side(sa_c, outer_center, outer_angle)
                s_pl = outer_side(pol_c, outer_center, outer_angle)
                sa_c = pol_c if s_pl < s_sa else sa_c
        except Exception:
            sa_c = st

        side = outer_side(sa_c, outer_center, outer_angle)
        if side < best_side:
            best_side = side
            best_centers = sa_c.copy()

    # Final validation; if invalid, nudge by slightly enlarging is not allowed,
    # so we trust the SA (which minimizes side with soft penalty). Ensure a valid
    # fallback exists.
    if best_centers is None:
        best_centers = init_hex_lattice()
        best_side = outer_side(best_centers, outer_center, outer_angle)

    # If the found packing is marginally invalid, do one last tight polish.
    gap = validity_gap(best_centers, outer_center, best_side, outer_angle)
    if gap < -1e-7 and remaining() > 3:
        try:
            pc = polish(best_centers, outer_center, outer_angle, max(1.0, remaining() - 1))
            ps = outer_side(pc, outer_center, outer_angle)
            if ps < best_side:
                best_centers, best_side = pc, ps
        except Exception:
            pass

    inner_list = [[float(x), float(y), float(a)] for (x, y, a) in best_centers]
    return inner_list, (float(outer_center[0]), float(outer_center[1])), float(best_side), float(outer_angle)
