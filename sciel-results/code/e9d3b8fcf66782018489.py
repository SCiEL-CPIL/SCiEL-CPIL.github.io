# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, least_squares

# ---------- geometry helpers ----------
def _hex_vertices(cx, cy, s, angle_deg):
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _all_vertices(positions, angles, side=1.0):
    n = len(positions)
    base_angles = np.radians(angles)[:, None] + np.radians(60) * np.arange(6)[None, :]
    verts = np.zeros((n, 6, 2))
    verts[:, :, 0] = positions[:, 0:1] + side * np.cos(base_angles)
    verts[:, :, 1] = positions[:, 1:2] + side * np.sin(base_angles)
    return verts

def _separated(positions, angles, tol=1e-6):
    n = len(positions)
    all_verts = _all_vertices(positions, angles)
    base = np.radians(angles)[:, None] + np.radians(30) + np.radians(60) * np.arange(6)[None, :]
    hex_normals = np.stack([np.cos(base), np.sin(base)], axis=2)
    for i in range(n):
        vi = all_verts[i]
        ni = hex_normals[i]
        for j in range(i + 1, n):
            vj = all_verts[j]
            nj = hex_normals[j]
            normals = np.vstack([ni, nj])
            proj_i = vi @ normals.T
            proj_j = vj @ normals.T
            if np.any(np.max(proj_i, axis=0) < np.min(proj_j, axis=0) - tol) \
               or np.any(np.max(proj_j, axis=0) < np.min(proj_i, axis=0) - tol):
                continue
            return False
    return True

def _enclosing_side(positions, angles, outer_angle=0.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(outer_angle + 30 + 60 * i)),
                         np.sin(np.radians(outer_angle + 30 + 60 * i))] for i in range(6)])
    projs = all_verts @ normals.T
    return np.max(projs) / (np.sqrt(3) / 2)

def _soft_max(x, beta=100.0):
    m = np.max(x)
    return m + np.log(np.mean(np.exp(x - m))) / beta

def _enclosing_side_soft(positions, angles, outer_angle=0.0, beta=100.0):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    normals = np.array([[np.cos(np.radians(outer_angle + 30 + 60 * i)),
                         np.sin(np.radians(outer_angle + 30 + 60 * i))] for i in range(6)])
    projs = all_verts @ normals.T
    return _soft_max(projs, beta) / (np.sqrt(3) / 2)

def _contained(positions, angles, side, outer_angle=0.0, tol=-1e-6):
    all_verts = _all_vertices(positions, angles).reshape(-1, 2)
    outer = _hex_vertices(0.0, 0.0, side, outer_angle)
    for i in range(6):
        p1 = outer[i]
        p2 = outer[(i + 1) % 6]
        e = p2 - p1
        v = all_verts - p1
        cross = e[0] * v[:, 1] - e[1] * v[:, 0]
        if np.min(cross) < tol:
            return False
    return True

# ---------- contact-based topology representation ----------
# Topology: hex 0 = center, 1-6 = ring, 7-10 = interstices
# Each hex i has a neighbor j with a contact edge. We parameterize by the
# contact angle (which edge of i touches which edge of j) and the relative rotation.
# Free variables: 11 positions (22) + 11 angles (11) = 33, but we constrain via contacts.

def _contact_residuals(x, topology):
    """
    x: [33] = 11 positions (22) + 11 angles (11)
    topology: list of (i, j, edge_i, edge_j) contact pairs
    Returns residuals: for each contact, the gap between the two touching edges
    (should be 0 for exact tangency, > 0 for separated)
    """
    pos = x[:22].reshape(11, 2)
    ang = x[22:]
    all_verts = _all_vertices(pos, ang)
    residuals = []
    for (i, j, ei, ej) in topology:
        # Edge ei of hex i: vertices ei and (ei+1)%6
        vi1 = all_verts[i, ei]
        vi2 = all_verts[i, (ei + 1) % 6]
        # Edge ej of hex j
        vj1 = all_verts[j, ej]
        vj2 = all_verts[j, (ej + 1) % 6]
        # Midpoints
        mi = (vi1 + vi2) / 2
        mj = (vj1 + vj2) / 2
        # Edge normals (outward from hex center)
        ci = pos[i]
        cj = pos[j]
        ni = mi - ci
        nj = mj - cj
        ni /= np.linalg.norm(ni) + 1e-15
        nj /= np.linalg.norm(nj) + 1e-15
        # The two edges should be parallel and opposite, with midpoints 2 apart along the normal
        # Residual 1: direction mismatch (edges should be parallel)
        cross_dir = ni[0] * nj[1] - ni[1] * nj[0]
        # Residual 2: distance between midpoints should be 2.0
        d = np.dot(mj - mi, ni)
        # Residual 3: perpendicular offset of midpoints
        perp = np.dot(mj - mi, np.array([-ni[1], ni[0]]))
        residuals.append(cross_dir * 10)
        residuals.append(d - 2.0)
        residuals.append(perp)
    return np.array(residuals)

def _build_topology():
    """
    Define contact topology for the 11-hexagon packing.
    Center (0) touches all 6 ring hexes (1-6).
    Each ring hex touches its two neighbors and possibly one interstice.
    Interstices (7-10) sit in the gaps between ring hexes.
    
    For a standard hexagonal packing of 11:
    - Hex 0 (center) touches hexes 1-6 on edges 0-5 respectively (depending on orientation)
    - Ring hex k touches ring hex k+1 (mod 6)
    - Interstices fill the triangular gaps
    
    Let's use a specific known topology:
    Hex 0 at origin, angle 0.
    Hex k (k=1..6) at 60*k degrees, touching hex 0.
    The edge of hex 0 facing hex k is edge k (if angle=0, edge 0 is at angle 0).
    Actually with angle=0, vertex 0 is at angle 0, so edge 0 is between vertices 0 and 1,
    centered at angle 30 degrees. Edge k is centered at 30+60k degrees.
    So hex 0's edge k faces direction 30+60k. Hex k is at direction 60k (for k=1..5) or 360 (k=6).
    Hmm, let me think more carefully.
    
    With angle_deg=0, vertex i is at angle 60i. Edge i is between vertices i and i+1,
    centered at angle 60i+30.
    
    For hex 0 at origin with angle 0:
    - Edge 0 center at 30 deg
    - Edge 1 center at 90 deg
    - ...
    - Edge 5 center at 330 deg
    
    If hex 1 is placed at direction 30 deg (touching edge 0 of hex 0), then hex 1's center is
    at distance 2 from origin in direction 30 deg. Hex 1's edge facing hex 0 is the one
    centered at 30+180=210 deg in hex 1's local frame. If hex 1 has angle 0, its edge 3
    is centered at 210 deg. So contact: (0, 1, 0, 3).
    
    Similarly:
    (0, 2, 1, 3) - hex 2 at 90 deg, touches edge 1 of hex 0
    (0, 3, 2, 3) - hex 3 at 150 deg, touches edge 2 of hex 0
    (0, 4, 3, 3) - hex 4 at 210 deg, touches edge 3 of hex 0
    (0, 5, 4, 3) - hex 5 at 270 deg, touches edge 4 of hex 0
    (0, 6, 5, 3) - hex 6 at 330 deg, touches edge 5 of hex 0
    
    Ring neighbors:
    (1, 2, 1, 5) - hex 1 and 2 are neighbors
    (2, 3, 1, 5)
    (3, 4, 1, 5)
    (4, 5, 1, 5)
    (5, 6, 1, 5)
    (6, 1, 1, 5)
    
    Interstices: 4 of them. In a 11-hexagon packing, the interstices are in the
    "valley" positions. Let's say:
    (1, 7, 0, 3) - hex 7 in the gap near hex 1
    (2, 8, 0, 3) - hex 8 in the gap near hex 2
    (4, 9, 0, 3) - hex 9 in the gap near hex 4
    (5, 10, 0, 3) - hex 10 in the gap near hex 5
    """
    topology = [
        (0, 1, 0, 3), (0, 2, 1, 3), (0, 3, 2, 3),
        (0, 4, 3, 3), (0, 5, 4, 3), (0, 6, 5, 3),
        (1, 2, 1, 5), (2, 3, 1, 5), (3, 4, 1, 5),
        (4, 5, 1, 5), (5, 6, 1, 5), (6, 1, 1, 5),
        (1, 7, 0, 3), (2, 8, 0, 3), (4, 9, 0, 3), (5, 10, 0, 3),
    ]
    return topology

def _initial_guess():
    """Generate a good initial guess for the 33-dim parameter vector."""
    pos = np.zeros((11, 2))
    ang = np.zeros(11)
    # Center
    pos[0] = [0, 0]
    ang[0] = 0
    # Ring hexes at 60-degree intervals, distance 2
    for k in range(6):
        a = np.radians(30 + 60 * k)
        pos[k + 1] = [2.0 * np.cos(a), 2.0 * np.sin(a)]
        ang[k + 1] = 0
    # Interstices at valley positions
    for k in range(4):
        a = np.radians(0 + 60 * k)
        pos[7 + k] = [3.464 * np.cos(a), 3.464 * np.sin(a)]  # 2*sqrt(3)
        ang[7 + k] = 0
    return np.concatenate([pos.flatten(), ang])

def _objective_full(x, topology, outer_angle=0.0):
    """
    Full objective: minimize enclosing side length while maintaining separation.
    """
    pos = x[:22].reshape(11, 2)
    ang = x[22:]
    if not _separated(pos, ang):
        return 1e6 + 0.01 * np.sum(pos**2)
    return _enclosing_side(pos, ang, outer_angle)

def _refine_full(positions, angles, deadline, rng, outer_angle=0.0):
    best_pos = positions.copy()
    best_ang = angles.copy()
    best_side = _enclosing_side(positions, angles, outer_angle)
    
    def obj(x):
        pos = x[:22].reshape(11, 2)
        ang = x[22:]
        if not _separated(pos, ang):
            return 1e6 + 0.01 * np.sum(pos**2)
        return _enclosing_side_soft(pos, ang, outer_angle, beta=100.0)
    
    x0 = np.concatenate([positions.flatten(), angles])
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-13,
                                'adaptive': True, 'maxfev': 200000})
        pos = res.x[:22].reshape(11, 2)
        ang = res.x[22:] % 360
        if _separated(pos, ang):
            s = _enclosing_side(pos, ang, outer_angle)
            if s < best_side:
                best_side = s
                best_pos = pos.copy()
                best_ang = ang.copy()
    except Exception:
        pass
    
    # Exact polish
    if time.time() < deadline - 30:
        def obj_exact(x):
            pos = x[:22].reshape(11, 2)
            ang = x[22:]
            if not _separated(pos, ang):
                return 1e6 + 0.01 * np.sum(pos**2)
            return _enclosing_side(pos, ang, outer_angle)
        
        x1 = np.concatenate([best_pos.flatten(), best_ang])
        try:
            res = minimize(obj_exact, x1, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-13,
                                    'adaptive': True, 'maxfev': 200000})
            pos = res.x[:22].reshape(11, 2)
            ang = res.x[22:] % 360
            if _separated(pos, ang):
                s = _enclosing_side(pos, ang, outer_angle)
                if s < best_side:
                    best_side = s
                    best_pos = pos.copy()
                    best_ang = ang.copy()
        except Exception:
            pass
    
    return best_pos, best_ang, best_side

def _best_rotation_joint(positions, angles):
    best_side = _enclosing_side(positions, angles, 0.0)
    best_theta = 0.0
    best_phi = 0.0
    
    def _eval(theta, phi):
        c = np.cos(np.radians(theta))
        s = np.sin(np.radians(theta))
        rot = np.array([[c, -s], [s, c]])
        p_rot = positions @ rot.T
        a_rot = (angles + theta) % 360
        return _enclosing_side(p_rot, a_rot, phi)
    
    for theta in np.linspace(0, 60, 31):
        for phi in np.linspace(0, 60, 31):
            s = _eval(theta, phi)
            if s < best_side:
                best_side = s
                best_theta = theta
                best_phi = phi
    
    for theta in np.linspace(max(0, best_theta - 5), min(60, best_theta + 5), 41):
        for phi in np.linspace(max(0, best_phi - 5), min(60, best_phi + 5), 41):
            s = _eval(theta, phi)
            if s < best_side:
                best_side = s
                best_theta = theta
                best_phi = phi
    
    try:
        res = minimize(
            lambda v: _eval(v[0], v[1]),
            [best_theta, best_phi],
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                     'adaptive': True, 'maxfev': 100000}
        )
        if res.fun < best_side:
            best_side = res.fun
            best_theta = res.x[0] % 60
            best_phi = res.x[1] % 60
    except Exception:
        pass
    
    c = np.cos(np.radians(best_theta))
    s = np.sin(np.radians(best_theta))
    rot = np.array([[c, -s], [s, c]])
    p_rot = positions @ rot.T
    a_rot = (angles + best_theta) % 360
    return p_rot, a_rot, best_phi, best_side

# ---------- main entry point ----------
def hexagon_packing_11():
    start = time.time()
    deadline = start + 580.0
    rng = np.random.default_rng(42)
    
    best_pos = None
    best_ang = None
    best_side = float('inf')
    best_phi = 0.0
    
    def consider(pos, ang):
        nonlocal best_pos, best_ang, best_side
        if not _separated(pos, ang):
            return False
        s = _enclosing_side(pos, ang, 0.0)
        if s < best_side:
            best_side = s
            best_pos = pos.copy()
            best_ang = ang.copy()
            return True
        return False
    
    # --- Phase A: Contact-constrained initial guess ---
    x0 = _initial_guess()
    pos0 = x0[:22].reshape(11, 2)
    ang0 = x0[22:]
    consider(pos0, ang0)
    
    # --- Phase B: Local refinement from initial guess ---
    pos1, ang1, s1 = _refine_full(pos0, ang0, deadline, rng, outer_angle=0.0)
    if s1 < best_side:
        best_side = s1
        best_pos = pos1.copy()
        best_ang = ang1.copy()
    
    # --- Phase C: Multi-start with perturbed initial guesses ---
    n_starts = 0
    while time.time() < deadline - 200 and n_starts < 50:
        # Perturb the initial guess
        scale = 0.1 * (1 + 0.05 * n_starts)
        pos_p = pos0 + rng.normal(0, scale, (11, 2))
        ang_p = (ang0 + rng.normal(0, scale * 5, 11)) % 360
        consider(pos_p, ang_p)
        
        pos2, ang2, s2 = _refine_full(pos_p, ang_p, deadline, rng, outer_angle=0.0)
        if s2 < best_side:
            best_side = s2
            best_pos = pos2.copy()
            best_ang = ang2.copy()
        n_starts += 1
    
    # --- Phase D: Try different topologies / interstice placements ---
    if time.time() < deadline - 100:
        for interstice_config in [(0, 1, 2, 3), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4)]:
            pos_t = pos0.copy()
            ang_t = ang0.copy()
            for idx, k in enumerate(interstice_config):
                a = np.radians(0 + 60 * k)
                pos_t[7 + idx] = [3.464 * np.cos(a), 3.464 * np.sin(a)]
                ang_t[7 + idx] = 0
            consider(pos_t, ang_t)
            pos3, ang3, s3 = _refine_full(pos_t, ang_t, deadline, rng, outer_angle=0.0)
            if s3 < best_side:
                best_side = s3
                best_pos = pos3.copy()
                best_ang = ang3.copy()
    
    # --- Fallback ---
    if best_pos is None:
        pos, ang = pos0, ang0
        best_pos, best_ang, best_side = pos, ang, _enclosing_side(pos, ang, 0.0)
    
    # --- Phase E: Polish the best ---
    if time.time() < deadline - 80:
        pos4, ang4, s4 = _refine_full(best_pos, best_ang, deadline, rng, outer_angle=0.0)
        if s4 < best_side:
            best_pos, best_ang, best_side = pos4, ang4, s4
    
    # --- Phase F: Joint rotation + outer angle ---
    if time.time() < deadline - 40:
        pos_r, ang_r, phi_r, s_r = _best_rotation_joint(best_pos, best_ang)
        if s_r < best_side:
            best_pos, best_ang, best_side, best_phi = pos_r, ang_r, phi_r, s_r
        if time.time() < deadline - 25:
            pos5, ang5, s5 = _refine_full(best_pos, best_ang, deadline, rng, outer_angle=best_phi)
            if s5 < best_side:
                best_pos, best_ang, best_side = pos5, ang5, s5
    
    # --- Final verification ---
    final_side = _enclosing_side(best_pos, best_ang, best_phi)
    if not _separated(best_pos, best_ang):
        pos, ang = pos0, ang0
        best_pos, best_ang, best_phi = pos, ang, 0.0
        final_side = _enclosing_side(pos, ang, 0.0)
    
    if not _contained(best_pos, best_ang, final_side, best_phi):
        final_side *= 1.001
    
    inner = np.column_stack([best_pos, best_ang])
    outer = np.array([0.0, 0.0, best_phi])
    return inner, outer, float(final_side)

# EVOLVE-BLOCK-END
