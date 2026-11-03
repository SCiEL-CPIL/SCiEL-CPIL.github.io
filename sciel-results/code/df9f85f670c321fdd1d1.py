import numpy as np
from scipy.optimize import linprog
import time

N = 11

def _hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon of side s centered at (cx,cy), rotated by angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + np.pi / 3 * k
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def _hex_lattice_positions():
    """Return center positions for a hexagonal cluster of radius 1 (13 sites), then drop 2 opposite corners to get 11."""
    # axial coordinates (q, r) for hex grid, radius 1
    axial = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if abs(q + r) <= 1:
                axial.append((q, r))
    # convert axial to cartesian (pointy-top, unit side)
    pos = []
    for q, r in axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pos.append((x, y))
    # that's 19 positions for radius 2; for radius 1 we want 7. Let me redo.
    # radius 1 hex cluster: 7 cells (center + 6 neighbors)
    axial = []
    for q in range(-1, 2):
        for r in range(-1, 2):
            if max(abs(q), abs(r), abs(q + r)) <= 1:
                axial.append((q, r))
    pos = []
    for q, r in axial:
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pos.append((x, y))
    # 7 positions. We need 11, so add a partial second ring.
    # Second ring has 12 cells. Add 4 of them (two opposite pairs) to make 11.
    # Actually, let's just use a simpler arrangement: a 3x3 grid of hexagons in a rhombus, or
    # a hexagonal number pattern. Let me use: center, 6 neighbors, and 4 more from the next ring
    # placed to keep symmetry.
    # For simplicity and symmetry, use: center + 6 ring1 + 4 ring2 (at angles 0,90,180,270 equivalent)
    # But hex grid doesn't have 90-degree symmetry. Let me just use a rectangular-ish cluster.
    
    # Simpler: use a 3x3 grid of hexagons in a "brick wall" pattern, that's 9.
    # For 11, use 3 rows: 4, 3, 4 = 11.
    positions = []
    # Row 0 (top): 4 hexagons
    for i in range(4):
        positions.append((i * np.sqrt(3), 0.0))
    # Row 1 (middle): 3 hexagons, offset by sqrt(3)/2
    for i in range(3):
        positions.append((i * np.sqrt(3) + np.sqrt(3) / 2, -1.5))
    # Row 2 (bottom): 4 hexagons
    for i in range(4):
        positions.append((i * np.sqrt(3), -3.0))
    return positions

def construct():
    start_time = time.time()
    DEADLINE = 130.0  # safety margin
    
    # Generate inner hexagon positions
    positions = _hex_lattice_positions()
    n_inner = len(positions)
    
    # Inner hexagons: side 1, angle 0, at each position
    inner = []
    for (px, py) in positions:
        inner.append([px, py, 0.0])
    
    # Compute all vertices of all inner hexagons
    all_verts = []
    for (px, py, ang) in inner:
        verts = _hexagon_vertices(px, py, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)  # shape (6*n_inner, 2)
    
    # Now optimize the outer hexagon: center (cx, cy), side s, angle theta
    # Containment: for each inner vertex v, it must be inside the outer hexagon.
    # A point is inside a regular hexagon of side s centered at (cx,cy) with angle theta
    # if and only if for each of the 6 faces, the signed distance is >= 0.
    # The 6 face normals are at angles theta + pi/3*k for k=0..5.
    # For a regular hexagon of side s, the apothem (inradius) is s * sqrt(3)/2.
    # A point p is inside iff for all k: (p - c) . n_k <= s * sqrt(3)/2, where n_k are unit normals.
    
    # This is nonlinear in s and theta. Let's fix theta=0 for simplicity (the inner arrangement
    # has a natural orientation) and optimize s and center via linear programming.
    
    # With theta = 0, the 6 face normals are at angles 0, 60, 120, 180, 240, 300 degrees.
    normals = []
    for k in range(6):
        theta = np.pi / 3 * k
        normals.append((np.cos(theta), np.sin(theta)))
    normals = np.array(normals)  # shape (6, 2)
    
    # For each vertex v_i and each normal n_j:
    # (v_i - c) . n_j <= s * sqrt(3)/2
    # => -v_i . n_j - c . n_j <= s * sqrt(3)/2
    # => -n_j . c - (sqrt(3)/2) * s <= v_i . n_j
    
    # Variables: [cx, cy, s]
    # Minimize: s (i.e., [0, 0, 1])
    
    m = len(all_verts)
    n_vars = 3
    
    A_ub = np.zeros((6 * m, n_vars))
    b_ub = np.zeros(6 * m)
    
    for j in range(6):
        nj = normals[j]
        for i in range(m):
            vi = all_verts[i]
            row = 6 * i + j
            A_ub[row, 0] = -nj[0]
            A_ub[row, 1] = -nj[1]
            A_ub[row, 2] = -np.sqrt(3) / 2
            b_ub[row] = np.dot(vi, nj)
    
    # s >= 0
    A_ub = np.vstack([A_ub, [[0, 0, -1]]])
    b_ub = np.append(b_ub, 0)
    
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Solve LP
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
        if res.success:
            cx_opt, cy_opt, s_opt = res.x
            # Ensure s is reasonable
            if s_opt < 0:
                s_opt = 0.0
        else:
            # Fallback: compute bounding box based approach
            # Find the max extent
            cx_opt, cy_opt = 0.0, 0.0
            s_opt = 10.0
    except Exception:
        cx_opt, cy_opt = 0.0, 0.0
        s_opt = 10.0
    
    # Recompute a safe side length: find the minimum s such that all vertices are inside
    # by checking each vertex against the hexagon with the optimized center
    def _required_side(cx, cy, verts, theta=0.0):
        apothem_factor = np.sqrt(3) / 2
        max_s = 0.0
        for k in range(6):
            theta_k = theta + np.pi / 3 * k
            nj = np.array([np.cos(theta_k), np.sin(theta_k)])
            for v in verts:
                d = np.dot(v - np.array([cx, cy]), nj)
                s_needed = d / apothem_factor
                if s_needed > max_s:
                    max_s = s_needed
        return max(0.0, max_s)
    
    s_safe = _required_side(cx_opt, cy_opt, all_verts)
    s_final = max(s_opt, s_safe)
    
    # Check time
    if time.time() - start_time > DEADLINE:
        # Return a safe default
        return inner, (0.0, 0.0), 50.0, 0.0
    
    return inner, (cx_opt, cy_opt), s_final, 0.0
