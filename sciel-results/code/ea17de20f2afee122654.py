# EVOLVE-BLOCK-START
import numpy as np
import time

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _all_vertices(centers, angles_deg, side=1.0):
    """(12,72,2) vertices for 12 hexagons."""
    n = len(centers)
    a = np.radians(angles_deg)
    base = a + np.arange(6) * np.pi / 3.0
    cos = np.cos(base)[:, None] * np.ones((n, 1))   # (n,6)
    sin = np.sin(base)[:, None] * np.ones((n, 1))   # (n,6)
    cos = cos * np.ones((n, 1))  # broadcast n
    # simpler explicit
    c = np.cos(a[:, None] + np.arange(6)[None, :] * np.pi / 3.0)
    s = np.sin(a[:, None] + np.arange(6)[None, :] * np.pi / 3.0)
    vx = centers[:, 0:1] + side * c
    vy = centers[:, 1:2] + side * s
    return np.stack([vx, vy], axis=-1)  # (n,6,2)

def _min_outer_side(verts):
    """verts: (72,2). Min side of outer hex (center origin, angle 0) containing all."""
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = verts @ m.T            # (72,6)
    max_dot = np.max(dots, axis=0)  # (6,)
    g = float(np.max(max_dot))
    if g < 0:
        g = 0.0
    return (2.0 / np.sqrt(3.0)) * g

def _soft_min_outer_side(verts, temp=0.05):
    """Smooth surrogate: soft-max over the 6 directions."""
    k = np.arange(6)
    angles = (k + 0.5) * np.pi / 3
    m = np.column_stack([np.cos(angles), np.sin(angles)])
    dots = verts @ m.T            # (72,6)
    # soft max over axis 0 (the 72 vertices) for each direction
    dmax = np.max(dots, axis=0)
    scaled = dots / temp
    scaled -= np.max(scaled, axis=0, keepdims=True)
    exp = np.exp(scaled)
    smax = temp * (np.log(np.sum(exp, axis=0)) + np.max(scaled, axis=0))
    g = float(np.max(smax))
    if g < 0:
        g = 0.0
    return (2.0 / np.sqrt(3.0)) * g

def _overlap_penalty(centers, angles_deg, side=1.0):
    """Vectorized SAT. Returns sum over pairs of max(0, overlap_margin)."""
    n = len(centers)
    verts = _all_vertices(centers, angles_deg, side)  # (n,6,2)
    # edge normals: for each hex, 6 edges
    # edge e: v[e+1]-v[e]; normal (ny, -nx)
    v = verts
    e = np.roll(v, -1, axis=1) - v          # (n,6,2)
    normals = np.stack([e[..., 1], -e[..., 0]], axis=-1)  # (n,6,2)
    # normalize normals
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normals = normals / norms

    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            # project both hexes onto all 12 normals
            n1 = normals[i]   # (6,2)
            n2 = normals[j]   # (6,2)
            all_n = np.vstack([n1, n2])  # (12,2)
            proj1 = v[i] @ all_n.T       # (6,12)
            proj2 = v[j] @ all_n.T       # (6,12)
            min1, max1 = np.min(proj1, axis=0), np.max(proj1, axis=0)
            min2, max2 = np.min(proj2, axis=0), np.max(proj2, axis=0)
            # overlap margin = max(0, min(max1-min2, max2-min1)) per axis; if any axis separates, overlap=0
            # separation on axis a: max1[a] < min2[a] or max2[a] < min1[a]
            sep = np.maximum(max1 - min2, max2 - min1)  # positive = overlap extent on that axis
            # if all axes have overlap extent >0, hexes overlap; the "amount" ~ min over axes of sep
            # use min sep as overlap measure; if any sep<=0 they are separated
            min_sep = float(np.min(sep))
            if min_sep > 0:
                pen += min_sep
    return pen

def _check_nonoverlap(centers, angles_deg, side=1.0, tol=1e-6):
    n = len(centers)
    verts = _all_vertices(centers, angles_deg, side)
    v = verts
    e = np.roll(v, -1, axis=1) - v
    normals = np.stack([e[..., 1], -e[..., 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normals = normals / norms
    for i in range(n):
        for j in range(i + 1, n):
            all_n = np.vstack([normals[i], normals[j]])
            proj1 = v[i] @ all_n.T
            proj2 = v[j] @ all_n.T
            min1, max1 = np.min(proj1, axis=0), np.max(proj1, axis=0)
            min2, max2 = np.min(proj2, axis=0), np.max(proj2, axis=0)
            sep = np.maximum(max1 - min2, max2 - min1)
            # if any axis separates (sep <= tol), they're apart
            if np.any(sep <= tol):
                continue
            else:
                return False
    return True

def _initial_honeycomb():
    d = np.sqrt(3.0)
    positions = []
    angles = []
    positions.append([0.0, 0.0]); angles.append(0.0)
    for k in range(6):
        angle = k * np.pi / 3
        positions.append([d * np.cos(angle), d * np.sin(angle)]); angles.append(0.0)
    d2 = 2.0 * d
    for k in range(5):
        angle = (k * 72 + 30) * np.pi / 180
        positions.append([d2 * np.cos(angle), d2 * np.sin(angle)]); angles.append(0.0)
    return np.array(positions), np.array(angles)

def _initial_ring():
    """Two concentric rings: 6 inner, 6 outer (12 total)."""
    d = np.sqrt(3.0)
    positions = []
    angles = []
    for k in range(6):
        angle = k * np.pi / 3
        positions.append([d * np.cos(angle), d * np.sin(angle)]); angles.append(0.0)
    d2 = 2.0 * d
    for k in range(6):
        angle = (k * 60 + 30) * np.pi / 180
        positions.append([d2 * np.cos(angle), d2 * np.sin(angle)]); angles.append(0.0)
    return np.array(positions), np.array(angles)

def _initial_linear():
    """A more spread arrangement: 3x4 grid-like."""
    d = np.sqrt(3.0)
    positions = []
    angles = []
    # 4 rows of 3
    for r in range(4):
        for c in range(3):
            x = c * 2.0 + (r % 2) * 1.0
            y = r * d * 0.9
            positions.append([x, y]); angles.append(0.0)
    return np.array(positions), np.array(angles)

def _objective(params, side=1.0, use_smooth=True, temp=0.05):
    params_2d = params.reshape(12, 3)
    centers = params_2d[:, :2]
    angles_deg = params_2d[:, 2]
    all_verts = _all_vertices(centers, angles_deg, side).reshape(72, 2)
    if use_smooth:
        s = _soft_min_outer_side(all_verts, temp)
    else:
        s = _min_outer_side(all_verts)
    pen = _overlap_penalty(centers, angles_deg, side)
    return s + 100.0 * pen

def _anneal(start_params, deadline, seed, T0=0.5, cooling=0.9995, step0=0.3,
            use_smooth=True, temp=0.05):
    np.random.seed(seed)
    current_params = start_params.copy()
    current_obj = _objective(current_params, use_smooth=use_smooth, temp=temp)
    best_params = current_params.copy()
    best_obj = current_obj
    T = T0
    step = step0
    while time.time() < deadline:
        new_params = current_params + np.random.randn(36) * step
        obj = _objective(new_params, use_smooth=use_smooth, temp=temp)
        delta = obj - current_obj
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-9)):
            current_params = new_params
            current_obj = obj
            if obj < best_obj:
                best_obj = obj
                best_params = new_params.copy()
        T *= cooling
        if step > 0.01:
            step *= 0.9999
    return best_params

def _polish(params, deadline):
    """High-precision local search with tight steps, exact objective."""
    np.random.seed(12345)
    best_params = params.copy()
    best_s = _min_outer_side(_all_vertices(params.reshape(12,3)[:,:2], params.reshape(12,3)[:,2]).reshape(72,2))
    step = 0.05
    while time.time() < deadline and step > 1e-5:
        improved = False
        for _ in range(2000):
            if time.time() > deadline:
                break
            trial = best_params + np.random.randn(36) * step
            c = trial.reshape(12,3)[:, :2]
            a = trial.reshape(12,3)[:, 2]
            if not _check_nonoverlap(c, a):
                continue
            s = _min_outer_side(_all_vertices(c, a).reshape(72,2))
            if s < best_s - 1e-12:
                best_s = s
                best_params = trial
                improved = True
        if not improved:
            step *= 0.5
    return best_params, best_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 560.0  # safety margin before 630s kill

    inits = [_initial_honeycomb(), _initial_ring(), _initial_linear()]
    best_params = None
    best_s = np.inf

    # Phase 1: multi-start annealing with smooth surrogate, split time
    per_start = (deadline - time.time()) / len(inits)
    for idx, (centers, angles) in enumerate(inits):
        params = np.column_stack([centers, angles]).flatten()
        # quick fix if overlapping
        if not _check_nonoverlap(centers, angles):
            scale = 1.1
            while not _check_nonoverlap(centers * scale, angles) and scale < 3.0:
                scale += 0.05
            centers = centers * scale
            params = np.column_stack([centers, angles]).flatten()
        local_deadline = start_time + 560.0 - (len(inits) - idx) * 20.0
        res = _anneal(params, local_deadline, seed=100 + idx, use_smooth=True, temp=0.05)
        c = res.reshape(12,3)[:, :2]
        a = res.reshape(12,3)[:, 2]
        if _check_nonoverlap(c, a):
            s = _min_outer_side(_all_vertices(c, a).reshape(72,2))
            if s < best_s:
                best_s = s
                best_params = res.copy()

    if best_params is None:
        centers, angles = _initial_honeycomb()
        best_params = np.column_stack([centers, angles]).flatten()
        best_s = _min_outer_side(_all_vertices(centers, angles).reshape(72,2))

    # Phase 2: polish with exact objective
    best_params, best_s = _polish(best_params, deadline)

    centers_final = best_params.reshape(12, 3)[:, :2]
    angles_final = best_params.reshape(12, 3)[:, 2]

    # center on outer hex center (origin)
    centroid = np.mean(centers_final, axis=0)
    centers_final = centers_final - centroid

    # final validation
    if not _check_nonoverlap(centers_final, angles_final):
        centers_final, angles_final = _initial_honeycomb()
    all_verts_final = _all_vertices(centers_final, angles_final).reshape(72, 2)
    s_final = _min_outer_side(all_verts_final)

    inner_hex_data = np.column_stack([centers_final, angles_final])
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, s_final

# EVOLVE-BLOCK-END
