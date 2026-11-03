import math
import time
import numpy as np

_N = 11
_APOTHEM = math.sqrt(3) / 2.0
_START = time.time()
_DEADLINE = 144.0


def _vertex_offsets(rot):
    """Unit hexagon vertex offsets for a given rotation (deg)."""
    a = math.radians(rot)
    c, s = math.cos(a), math.sin(a)
    out = np.empty((6, 2))
    for k in range(6):
        t = a + k * math.pi / 3.0
        out[k, 0] = math.cos(t)
        out[k, 1] = math.sin(t)
    return out


def _hex_vertices(x, y, rot):
    return _vertex_offsets(rot) + np.array([x, y])


def _outer_normals(cx, cy, R, beta):
    """Six outward normals of the outer hexagon."""
    out = np.empty((6, 2))
    for k in range(6):
        t = beta + math.pi / 6.0 + k * math.pi / 3.0
        out[k, 0] = math.cos(t)
        out[k, 1] = math.sin(t)
    return out


def _objective(params):
    """
    params: [x_0..x_10, y_0..y_10, r_0..r_10 (radians), cx, cy, R, beta]
    """
    if time.time() > _DEADLINE:
        return 1e6
    n = _N
    xs = params[0:n]
    ys = params[n:2*n]
    rts = params[2*n:3*n]
    cx, cy = params[3*n], params[3*n+1]
    R = params[3*n+2]
    beta = params[3*n+3]
    if R < 0.5:
        R = 0.5

    # Build all 66 vertex arrays (N x 6 x 2)
    verts = np.empty((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(xs[i], ys[i], rts[i])

    # Outer normals
    normals = np.empty((6, 2))
    for k in range(6):
        t = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals[k, 0] = math.cos(t)
        normals[k, 1] = math.sin(t)

    # Outer wall distance for each vertex: d_k(v) = n_k . (v - c) - R
    # (66 x 6)
    d_outer = np.empty((n*6, 6))
    idx = 0
    for i in range(n):
        for j in range(6):
            v = verts[i, j]
            for k in range(6):
                d_outer[idx, k] = normals[k, 0]*(v[0]-cx) + normals[k, 1]*(v[1]-cy) - R
            idx += 1

    # Pairwise hexagon non-overlap: for each pair (i, j), i<j,
    # check that for every edge of hex i, all vertices of hex j are on the correct side (>= 0).
    # We use the minimum clearance: min over all (edge of i, vertex of j) of n_edge . (v_j - v_edge_origin)
    # If any is negative, overlap. We penalize the max of (-clearance).

    pair_penalty = 0.0
    for i in range(n):
        vi = verts[i]  # (6,2)
        for j in range(i+1, n):
            vj = verts[j]  # (6,2)
            # Edges of hex i: for each vertex vi[j], edge direction to vi[(j+1)%6]
            # Outward normal for hex i edge j: perpendicular to (vi[j+1]-vi[j]) rotated appropriately
            # For a regular hexagon with vertices in CCW order, outward normal at vertex j
            # (between vertex j and j+1) is the angle bisector outward.
            # Actually for a regular polygon, the outward normal to edge from vertex j to j+1
            # is the unit vector perpendicular to (v_{j+1} - v_j), pointing away from center.
            for e in range(6):
                v0 = vi[e]
                v1 = vi[(e+1) % 6]
                edge = v1 - v0
                edge_len = math.sqrt(edge[0]**2 + edge[1]**2)
                if edge_len < 1e-12:
                    continue
                # Perpendicular: rotate edge by -90 deg for outward normal (CCW polygon)
                nx = -edge[1] / edge_len
                ny = edge[0] / edge_len
                # Check all vertices of hex j
                for f in range(6):
                    vj_f = vj[f]
                    clearance = nx * (vj_f[0] - v0[0]) + ny * (vj_f[1] - v0[1])
                    if clearance < 0:
                        pair_penalty += (-clearance) ** 2

    # Outer containment penalty
    # For each vertex, the max over k of d_outer[idx, k] must be <= 0 (inside or on wall)
    max_outer = d_outer.max(axis=1)  # (66,)
    outer_penalty = np.sum(np.maximum(0, max_outer) ** 2)

    total = outer_penalty + 5.0 * pair_penalty + 0.1 * R
    return total


def construct():
    n = _N
    # Initial guess: hexagonal lattice arrangement
    # Use a compact cluster of 11 hexagons
    # Axial coordinates for a good starting cluster
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    init = np.empty(3*n + 4)
    for i, (q, r) in enumerate(axial):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        init[i] = x
        init[n+i] = y
        init[2*n+i] = 0.0  # rotation
    
    cx0 = float(np.mean(init[0:n]))
    cy0 = float(np.mean(init[n:2*n]))
    init[3*n] = cx0
    init[3*n+1] = cy0
    init[3*n+2] = 3.5  # initial R
    init[3*n+3] = 0.0  # initial beta

    t0 = time.time()
    best_params = init.copy()
    best_val = _objective(init)

    # L-BFGS-B optimization
    from scipy.optimize import minimize as sp_minimize
    res = sp_minimize(_objective, init, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
    if res.fun < best_val:
        best_val = res.fun
        best_params = res.x.copy()

    # Try multiple random initializations if time permits
    rng = np.random.default_rng(42)
    attempts = 0
    while time.time() < _DEADLINE - 15.0 and attempts < 20:
        # Generate a random but reasonable starting configuration
        # Start with a lattice and add small perturbations
        init2 = init.copy()
        init2[0:2*n] += rng.normal(0, 0.1, 2*n)
        init2[2*n:3*n] += rng.normal(0, 0.2, n)
        init2[3*n+2] = 3.0 + rng.uniform(-0.5, 0.5)
        init2[3*n+3] = rng.uniform(0, math.pi/6)
        
        t_start = time.time()
        res2 = sp_minimize(_objective, init2, method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if res2.fun < best_val:
            best_val = res2.fun
            best_params = res2.x.copy()
        attempts += 1
        if time.time() - t_start > 5.0:
            break

    # Extract best solution
    xs = best_params[0:n]
    ys = best_params[n:2*n]
    rts = best_params[2*n:3*n]
    cx, cy = best_params[3*n], best_params[3*n+1]
    R = best_params[3*n+2]
    beta = best_params[3*n+3]

    # Refine: use the actual minimum R needed
    # Compute the true minimum R for this configuration
    best_R = float('inf')
    best_beta = beta
    for b in np.linspace(0, math.pi/6, 12):
        normals = np.empty((6, 2))
        for k in range(6):
            t = b + math.pi / 6.0 + k * math.pi / 3.0
            normals[k, 0] = math.cos(t)
            normals[k, 1] = math.sin(t)
        
        # For each outer normal direction, find the max projection
        max_proj = 0.0
        for i in range(n):
            verts_i = _hex_vertices(xs[i], ys[i], rts[i])
            for j in range(6):
                v = verts_i[j]
                for k in range(6):
                    proj = normals[k, 0]*(v[0]-cx) + normals[k, 1]*(v[1]-cy)
                    if abs(proj) > max_proj:
                        max_proj = abs(proj)
        R_candidate = max_proj
        if R_candidate < best_R:
            best_R = R_candidate
            best_beta = b

    # Newton polish for center at best_beta
    normals = np.empty((6, 2))
    for k in range(6):
        t = best_beta + math.pi / 6.0 + k * math.pi / 3.0
        normals[k, 0] = math.cos(t)
        normals[k, 1] = math.sin(t)

    # Build all vertex positions
    all_verts = np.empty((n*6, 2))
    idx = 0
    for i in range(n):
        verts_i = _hex_vertices(xs[i], ys[i], rts[i])
        for j in range(6):
            all_verts[idx] = verts_i[j]
            idx += 1

    # Optimize center to minimize max projection
    cx_c, cy_c = cx, cy
    for _ in range(200):
        D = 0.0
        active_A = []
        active_b = []
        for k in range(6):
            nx, ny = normals[k]
            for m in range(n*6):
                proj = nx*(all_verts[m,0]-cx_c) + ny*(all_verts[m,1]-cy_c)
                if proj > D:
                    D = proj
                    active_A.append([nx, ny])
                    active_b.append(proj)
                if -proj > D:
                    D = -proj
                    active_A.append([-nx, -ny])
                    active_b.append(proj)
        
        if len(active_A) < 2:
            break
        A = np.array(active_A)
        b = np.array(active_b)
        try:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            if abs(sol[0]-cx_c) < 1e-14 and abs(sol[1]-cy_c) < 1e-14:
                break
            cx_c, cy_c = sol[0], sol[1]
        except:
            break

    # Recompute final R
    D_final = 0.0
    for k in range(6):
        nx, ny = normals[k]
        for m in range(n*6):
            proj = abs(nx*(all_verts[m,0]-cx_c) + ny*(all_verts[m,1]-cy_c))
            if proj > D_final:
                D_final = proj

    S = (D_final + _APOTHEM) / _APOTHEM
    beta_deg = math.degrees(best_beta)

    inner = []
    for i in range(n):
        inner.append([float(xs[i]), float(ys[i]), math.degrees(float(rts[i]))])
    center = [float(cx_c), float(cy_c)]
    side = float(S)
    rotation = float(beta_deg)

    return inner, center, side, rotation
