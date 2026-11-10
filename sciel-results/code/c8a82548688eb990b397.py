# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _min_outer_side(vertices):
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60 * k)), np.sin(np.radians(30 + 60 * k))]
        for k in range(6)
    ])
    dots = vertices @ outward_normals.T
    max_dot = np.max(dots, axis=1)
    apothem_min = np.max(max_dot)
    s_min = 2.0 * apothem_min / np.sqrt(3.0)
    return s_min

def _check_separation(pos1, ang1, pos2, ang2, side=1.0):
    v1 = _hex_vertices(pos1[0], pos1[1], side, ang1)
    v2 = _hex_vertices(pos2[0], pos2[1], side, ang2)
    normals = []
    for v in [v1, v2]:
        for i in range(6):
            edge = v[(i + 1) % 6] - v[i]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                normals.append(n)
    for n in normals:
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return True
    return False

def _is_valid(inner, outer, s):
    n = inner.shape[0]
    if n != 12:
        return False
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False
    # outer vertices distinct
    ov = _hex_vertices(outer[0], outer[1], s, outer[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(ov[i] - ov[j]) < 1e-9:
                return False
    # shift so outer center is origin
    oc = np.array([outer[0], outer[1]])
    pos = inner[:, :2] - oc
    angles = inner[:, 2]
    # check unit edge lengths
    for i in range(12):
        v = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
        for k in range(6):
            e = v[(k + 1) % 6] - v[k]
            if abs(np.linalg.norm(e) - 1.0) > 1e-8:
                return False
    # separation
    for a in range(12):
        for b in range(a + 1, 12):
            if not _check_separation(pos[a], angles[a], pos[b], angles[b], 1.0):
                return False
    # containment
    for i in range(12):
        v = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
        for k in range(6):
            p1 = ov[k]
            p2 = ov[(k + 1) % 6]
            for vtx in v:
                cross = (p2[0] - p1[0]) * (vtx[1] - p1[1]) - (p2[1] - p1[1]) * (vtx[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True

def _objective(params, n=12):
    """
    params: (48,) = [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11]
    Returns (s + big_penalty)
    """
    pos = params[0::3].reshape(n, 2)
    angles = params[1::3]  # degrees
    # vertices
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        all_verts[i * 6:(i + 1) * 6] = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
    # s
    s = _min_outer_side(all_verts)
    # overlap penalty: for each pair, compute min SAT overlap (positive = overlap)
    overlap_pen = 0.0
    for i in range(n):
        v1 = all_verts[i * 6:(i + 1) * 6]
        for j in range(i + 1, n):
            v2 = all_verts[j * 6:(j + 1) * 6]
            # quick reject
            if np.linalg.norm(pos[i] - pos[j]) > 2.05:
                continue
            normals = []
            for v in [v1, v2]:
                for k in range(6):
                    edge = v[(k + 1) % 6] - v[k]
                    nn = np.array([-edge[1], edge[0]])
                    nm = np.linalg.norm(nn)
                    if nm > 1e-12:
                        nn = nn / nm
                        normals.append(nn)
            min_ov = np.inf
            for nrm in normals:
                proj1 = v1 @ nrm
                proj2 = v2 @ nrm
                ov = min(np.max(proj1), np.max(proj2)) - max(np.min(proj1), np.min(proj2))
                if ov > 0:
                    min_ov = min(min_ov, ov)
            if min_ov < np.inf:
                overlap_pen += min_ov ** 2
    # containment penalty: for each inner vertex, check against outer edges (angle 0, center origin, side s)
    # Actually we compute s from the config, so we need to check containment against the computed s
    # The outer hexagon has center (0,0), angle 0, side s
    # Its edges: for each edge, the inward normal direction
    # A point p is inside if for all 6 edges, dot(p, outward_normal) <= apothem
    # apothem = s * sqrt(3) / 2
    apothem = s * np.sqrt(3.0) / 2.0
    outward_normals = np.array([
        [np.cos(np.radians(30 + 60 * k)), np.sin(np.radians(30 + 60 * k))]
        for k in range(6)
    ])
    contain_pen = 0.0
    for i in range(n):
        v = all_verts[i * 6:(i + 1) * 6]
        dots = v @ outward_normals.T  # (6, 6)
        excess = np.maximum(dots - apothem, 0.0)
        contain_pen += np.sum(excess ** 2)
    return s + 100.0 * overlap_pen + 100.0 * contain_pen

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 570.0  # safety margin
    n = 12
    best_obj = np.inf
    best_params = None

    # Generate diverse initial configurations
    rng = np.random.default_rng(42)
    inits = []
    # 1. Honeycomb grid
    hp = []
    for row in range(4):
        for col in range(3):
            x = col * 3.0
            y = row * np.sqrt(3)
            if row % 2 == 1:
                x += 1.5
            hp.append([x, y, 0.0])
    hp = np.array(hp)
    hp[:, :2] -= hp[:, :2].mean(axis=0)
    inits.append(hp.flatten())
    # 2. Tighter honeycomb
    hp2 = []
    for row in range(4):
        for col in range(3):
            x = col * 2.9
            y = row * np.sqrt(3) * 0.95
            if row % 2 == 1:
                x += 1.45
            hp2.append([x, y, 0.0])
    hp2 = np.array(hp2)
    hp2[:, :2] -= hp2[:, :2].mean(axis=0)
    inits.append(hp2.flatten())
    # 3. Ring + center
    ring = []
    ring.append([0, 0, 0.0])
    for k in range(6):
        a = np.radians(60 * k)
        ring.append([2.0 * np.cos(a), 2.0 * np.sin(a), 0.0])
    for k in range(6):
        a = np.radians(60 * k + 30)
        ring.append([3.5 * np.cos(a), 3.5 * np.sin(a), 0.0])
    ring = np.array(ring)
    ring[:, :2] -= ring[:, :2].mean(axis=0)
    inits.append(ring.flatten())
    # 4-8. Random perturbations of honeycomb
    for trial in range(5):
        base = hp.copy()
        base[:, :2] += rng.normal(0, 0.3, (n, 2))
        base[:, 2] += rng.normal(0, 10.0, n)
        inits.append(base.flatten())
    # 9-12. Random scatter
    for trial in range(4):
        p = rng.uniform(-3, 3, (n, 2))
        p -= p.mean(axis=0)
        a = rng.uniform(0, 60, n)
        params = np.zeros(3 * n)
        params[0::3] = p[:, 0]
        params[1::3] = p[:, 1]
        params[2::3] = a
        inits.append(params)

    for init in inits:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective, init, method='L-BFGS-B',
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000})
            obj = res.fun
            if obj < best_obj:
                best_obj = obj
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = inits[0].copy()

    # Decode
    pos = best_params[0::3].reshape(n, 2)
    angles = best_params[1::3]
    # Center at origin
    pos -= pos.mean(axis=0)
    inner = np.column_stack([pos, angles])
    outer = np.array([0.0, 0.0, 0.0])
    # Compute s
    all_verts = np.zeros((n * 6, 2))
    for i in range(n):
        all_verts[i * 6:(i + 1) * 6] = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
    s = _min_outer_side(all_verts)

    # If invalid, try to repair by scaling positions slightly
    if not _is_valid(inner, outer, s):
        # Try scaling down positions to ensure containment, and check separation
        for scale in [0.99, 0.98, 0.97, 0.96, 0.95]:
            pos2 = pos * scale
            inner2 = np.column_stack([pos2, angles])
            all_verts2 = np.zeros((n * 6, 2))
            for i in range(n):
                all_verts2[i * 6:(i + 1) * 6] = _hex_vertices(pos2[i, 0], pos2[i, 1], 1.0, angles[i])
            s2 = _min_outer_side(all_verts2)
            if _is_valid(inner2, outer, s2):
                inner = inner2
                s = s2
                break
        else:
            # Fallback: use a known valid configuration (honeycomb)
            hp = []
            for row in range(4):
                for col in range(3):
                    x = col * 3.0
                    y = row * np.sqrt(3)
                    if row % 2 == 1:
                        x += 1.5
                    hp.append([x, y, 0.0])
            hp = np.array(hp)
            hp[:, :2] -= hp[:, :2].mean(axis=0)
            inner = hp
            all_verts = np.zeros((n * 6, 2))
            for i in range(n):
                all_verts[i * 6:(i + 1) * 6] = _hex_vertices(hp[i, 0], hp[i, 1], 1.0, hp[i, 2])
            s = _min_outer_side(all_verts)

    return inner, outer, s

# EVOLVE-BLOCK-END
