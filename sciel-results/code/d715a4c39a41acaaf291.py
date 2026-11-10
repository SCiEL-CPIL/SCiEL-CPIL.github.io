# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    verts = np.zeros((n, 6, 2))
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _check_separation_fast(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    hex_normals = np.zeros((n, 6, 2))
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals[:, :, 0] = np.cos(base)
    hex_normals[:, :, 1] = np.sin(base)

    for i in range(n):
        for j in range(i + 1, n):
            vi = all_verts[i]
            vj = all_verts[j]
            normals = np.vstack([hex_normals[i], hex_normals[j]])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            max_i = np.max(proj_i, axis=0)
            min_i = np.min(proj_i, axis=0)
            max_j = np.max(proj_j, axis=0)
            min_j = np.min(proj_j, axis=0)
            if np.any(max_i < min_j - tol) or np.any(max_j < min_i - tol):
                continue
            return False
    return True

def _min_enclosing_side(positions, angles):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
    apothem_factor = np.sqrt(3) / 2
    projs = all_verts @ normals.T
    max_proj = np.max(projs)
    return max_proj / apothem_factor

def _check_containment_fast(positions, angles, side, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer_verts = _hex_vertices(0.0, 0.0, side, 0.0)
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

_NORMALS = np.array([[np.cos(np.radians(30 + 60 * i)), np.sin(np.radians(30 + 60 * i))] for i in range(6)])
_APO = np.sqrt(3) / 2

def _soft_max(x, beta):
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m))) / beta

def _objective(v, beta):
    pos = v[:22].reshape(11, 2)
    ang = v[22:]
    all_verts = _all_vertices(pos, ang).reshape(-1, 2)
    projs = all_verts @ _NORMALS.T
    side = _soft_max(projs, beta) / _APO
    n = 11
    verts = _all_vertices(pos, ang)
    base = np.radians(ang)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    normals = np.stack([np.cos(base), np.sin(base)], axis=2)  # (n,6,2)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            norm = np.vstack([normals[i], normals[j]])  # (12,2)
            pi = verts[i] @ norm.T
            pj = verts[j] @ norm.T
            mi = np.max(pi, axis=0); mni = np.min(pi, axis=0)
            mj = np.max(pj, axis=0); mnj = np.min(pj, axis=0)
            d1 = mni - mj   # i fully left of j -> want > 0
            d2 = mni - mi   # j fully left of i -> want > 0
            gap = np.minimum(d1, d2)
            bad = np.maximum(0.0, -gap)
            pen += np.sum(bad * bad)
    return side + 100.0 * pen

def _seed_honeycomb():
    pos = np.zeros((11, 2)); ang = np.zeros(11)
    pos[0] = [0, 0]
    for k in range(6):
        a = np.radians(30 + 60 * k)
        pos[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
    for k in range(4):
        a = np.radians(0 + 60 * k)
        pos[7 + k] = [3.5 * np.cos(a), 3.5 * np.sin(a)]
    return pos, ang

def _seed_ring10():
    pos = np.zeros((11, 2)); ang = np.zeros(11)
    pos[0] = [0, 0]
    for k in range(6):
        a = np.radians(30 + 60 * k)
        pos[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
    for k in range(4):
        a = np.radians(0 + 90 * k)
        pos[7 + k] = [3.6 * np.cos(a), 3.6 * np.sin(a)]
    return pos, ang

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 615.0

    best_positions, best_angles = None, None
    best_side = float('inf')

    def consider(pos, ang, beta):
        nonlocal best_positions, best_angles, best_side
        side = _min_enclosing_side(pos, ang)
        if not _check_separation_fast(pos, ang):
            return
        if side < best_side:
            best_side = side
            best_positions = pos.copy()
            best_angles = ang.copy()

    seeds = [_seed_honeycomb(), _seed_ring10()]

    # Phase 1: coarse optimization from each seed (anneal beta down)
    for pos0, ang0 in seeds:
        v0 = np.concatenate([pos0.ravel(), ang0])
        v = v0.copy()
        for beta in [25.0, 12.0, 6.0, 3.0]:
            if time.time() > deadline - 60:
                break
            try:
                res = minimize(_objective, v, args=(beta,), method='L-BFGS-B',
                               options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-8})
                if res.success or res.fun < _objective(v, beta):
                    v = res.x
            except Exception:
                pass
        pos = v[:22].reshape(11, 2); ang = v[22:]
        consider(pos, ang, 3.0)

    # Phase 2: multi-start refinement from best + random perturbations
    if best_positions is not None:
        starts = [np.concatenate([best_positions.ravel(), best_angles])]
        rng = np.random.default_rng(12345)
        for _ in range(10):
            d = rng.normal(0, 0.35, size=(11, 2))
            da = rng.uniform(-15, 15, size=11)
            p = best_positions + d
            a = (best_angles + da) % 360
            starts.append(np.concatenate([p.ravel(), a]))
        for v0 in starts:
            if time.time() > deadline - 40:
                break
            v = v0.copy()
            for beta in [6.0, 3.0, 1.5]:
                if time.time() > deadline - 40:
                    break
                try:
                    res = minimize(_objective, v, args=(beta,), method='L-BFGS-B',
                                   options={'maxiter': 400, 'ftol': 1e-15, 'gtol': 1e-9})
                    if res.success or res.fun < _objective(v, beta):
                        v = res.x
                except Exception:
                    pass
            pos = v[:22].reshape(11, 2); ang = v[22:]
            consider(pos, ang, 1.5)

    # Fallback
    if best_positions is None:
        pos, ang = _seed_honeycomb()
        best_positions, best_angles = pos, ang
        best_side = _min_enclosing_side(pos, ang)

    # Final verification / containment safety
    final_side = _min_enclosing_side(best_positions, best_angles)
    if not _check_separation_fast(best_positions, best_angles):
        pos, ang = _seed_honeycomb()
        best_positions, best_angles = pos, ang
        final_side = _min_enclosing_side(pos, ang)
    if not _check_containment_fast(best_positions, best_angles, final_side):
        final_side *= 1.001

    inner = np.column_stack([best_positions, best_angles])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(final_side)
    return inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
