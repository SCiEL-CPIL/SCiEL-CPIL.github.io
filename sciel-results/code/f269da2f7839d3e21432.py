# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(hex_verts):
    """6 outward unit edge normals for a hexagon given 6 ordered vertices."""
    normals = []
    centroid = hex_verts.mean(axis=0)
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def _build_inner_hexes(positions, angles):
    hexes = []
    for i in range(11):
        verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
        norms = _edge_normals(verts)
        hexes.append((verts, norms))
    return hexes


def _separated(v1, n1, v2, n2):
    """True if two hexagons are apart by > 1e-6 on at least one of their 12 edge normals."""
    all_normals = np.vstack([n1, n2])  # 12 x 2
    proj1 = v1 @ all_normals.T  # 6 x 12
    proj2 = v2 @ all_normals.T
    min1, max1 = proj1.min(axis=0), proj1.max(axis=0)
    min2, max2 = proj2.min(axis=0), proj2.max(axis=0)
    gaps = np.minimum(max1 - min2, max2 - min1)
    return bool(np.any(gaps > 1e-6))


def _all_separated(inner_hexes):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(inner_hexes[i][0], inner_hexes[i][1],
                              inner_hexes[j][0], inner_hexes[j][1]):
                return False
    return True


def _fits(S, angle_deg, inner_hexes):
    outer_verts = _hex_vertices(0.0, 0.0, S, angle_deg)
    for hex_verts, _ in inner_hexes:
        for p in hex_verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True


def _required_side(all_pts, angle_deg):
    """Smallest S such that all_pts fit a centered outer hexagon of side S at this angle."""
    outer_verts = _hex_vertices(0.0, 0.0, 1.0, angle_deg)  # unit side
    best = 1e-9
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        e = p2 - p1
        val = e[0] * (all_pts[:, 1] - p1[1]) - e[1] * (all_pts[:, 0] - p1[0])
        worst = float(val.min())
        if worst < 0.0:
            best = max(best, -worst)
    return best


def _min_outer_side(inner_hexes):
    """Minimum (S, angle) of a centered outer hexagon containing all inner hexagons."""
    all_pts = np.vstack([hv for hv, _ in inner_hexes])  # (66, 2)
    max_r = float(np.max(np.linalg.norm(all_pts, axis=1)))

    angles = np.arange(0.0, 60.0, 5.0)
    reqs = np.array([_required_side(all_pts, a) for a in angles])
    i0 = int(np.argmin(reqs))
    best_S = float(reqs[i0])
    best_angle = float(angles[i0])

    fine = best_angle + np.linspace(-5.0, 5.0, 41)
    for a in fine:
        r = _required_side(all_pts, a)
        if r < best_S:
            best_S = r
            best_angle = float(a)

    sl = max_r * 0.9
    sh = best_S
    for _ in range(30):
        sm = (sl + sh) / 2.0
        if _fits(sm, best_angle, inner_hexes):
            sh = sm
        else:
            sl = sm
    best_S = sh

    if not _fits(best_S, best_angle, inner_hexes):
        S = max_r
        for _ in range(60):
            if _fits(S, best_angle, inner_hexes):
                break
            S *= 1.05
        best_S = S

    return best_S, best_angle


def _seed_positions():
    s = 2.5 * np.sqrt(3) / 2.0
    return np.array([
        [0.0, 0.0],
        [-2.5, 0.0],
        [2.5, 0.0],
        [-1.25, s],
        [1.25, s],
        [-1.25, -s],
        [1.25, -s],
        [-3.75, s],
        [3.75, s],
        [-3.75, -s],
        [3.75, -s],
    ])


def _free_body(positions, angles, n_iter=4000, dt=0.006, k_rep=800.0,
               k_att=6.0, damp=0.90, max_step=0.08):
    """Rigid-body (translational) relaxation: repel overlapping hexagons, gently
    pull each toward the origin, with velocity damping. Returns improved positions."""
    pos = np.array(positions, dtype=float)
    vel = np.zeros_like(pos)
    n = len(pos)

    for it in range(n_iter):
        # per-pair repulsive forces from the separating-normal gap (negative => overlap)
        f = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                v1 = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
                v2 = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, angles[j])
                n1 = _edge_normals(v1)
                n2 = _edge_normals(v2)
                all_normals = np.vstack([n1, n2])  # 12 x 2
                proj1 = v1 @ all_normals.T
                proj2 = v2 @ all_normals.T
                gaps = np.minimum(proj1.max(axis=0) - proj2.min(axis=0),
                                  proj2.max(axis=0) - proj1.min(axis=0))
                worst = float(gaps.max())
                if worst < 0.0:  # overlapping: push apart along the separating normal
                    idx = int(np.argmin(gaps))
                    nrm = all_normals[idx]
                    # nrm points from hex j toward hex i (i.max < j.min means i is on -nrm side)
                    d = nrm
                    f[i] += k_rep * (-worst) * d
                    f[j] += k_rep * (-worst) * (-d)
        # gentle pull to origin (keeps the cluster compact and centered)
        f -= k_att * pos
        # integrate with damping and step clamp
        vel = damp * vel + dt * f
        step = vel * dt
        step = np.clip(step, -max_step, max_step)
        pos += step
    return pos


def _objective(params, deadline, safety):
    if time.time() > deadline - safety:
        return 1e10
    positions = params[:22].reshape(11, 2)
    angles = params[22:]
    inner_hexes = _build_inner_hexes(positions, angles)
    if not _all_separated(inner_hexes):
        return 1e10
    S, _ = _min_outer_side(inner_hexes)
    if not np.isfinite(S):
        return 1e10
    return S


def hexagon_packing_11():
    t_start = time.time()
    deadline = 600.0
    safety = 120.0

    seed = _seed_positions()
    angles0 = np.zeros(11)

    def _score(positions, angles):
        hexes = _build_inner_hexes(positions, angles)
        if not _all_separated(hexes):
            return None
        S, a = _min_outer_side(hexes)
        if not np.isfinite(S):
            return None
        return S, a

    # baseline from seed
    base = _score(seed, angles0)
    best_S = base[0]
    best_angle = base[1]
    best_pos = seed.copy()
    best_ang = angles0.copy()

    # ---- Structural: free-body discovery of a denser, non-symmetric packing ----
    if time.time() < deadline - safety:
        for iters in [4000, 8000]:
            if time.time() > deadline - safety:
                break
            sim = _free_body(seed, angles0, n_iter=iters)
            sc = _score(sim, angles0)
            if sc is not None and sc[0] < best_S:
                best_S, best_angle = sc
                best_pos, best_ang = sim.copy(), angles0.copy()

    # ---- Fast local polish (vectorized objective) from the best found ----
    if time.time() < deadline - safety:
        x0 = np.zeros(33)
        x0[:22] = best_pos.flatten()
        x0[22:] = best_ang

        try:
            res = minimize(_objective, x0, args=(deadline, safety),
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
            if res.fun < best_S and res.fun < 1e9:
                cand = res.x
                cpos = cand[:22].reshape(11, 2)
                can = cand[22:]
                sc = _score(cpos, can)
                if sc is not None and sc[0] < best_S:
                    best_S, best_angle = sc
                    best_pos, best_ang = cpos, can
        except Exception:
            pass

    # ---- Final verification; fall back to seed if anything is off ----
    final = _score(best_pos, best_ang)
    if final is None:
        best_pos, best_ang = seed.copy(), angles0.copy()
        best_S, best_angle = base

    inner_hex_data = np.column_stack([best_pos, best_ang])
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    outer_hex_side_length = float(best_S)
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
