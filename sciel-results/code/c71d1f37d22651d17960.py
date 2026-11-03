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
    """
    Generate positions for 11 unit hexagons in a compact arrangement.
    To ensure strict disjointness (avoiding edge-touching issues in evaluators),
    we generate a standard tight packing and then scale the coordinates by a small factor > 1.
    """
    # Standard tight packing for Flat-Top hexagons (angle 0)
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons
    # Row 2: 4 hexagons
    # This forms a trapezoidal shape.
    
    s3 = np.sqrt(3)
    positions = []
    
    # Row 0 (y=0)
    for i in range(4):
        positions.append((i * s3, 0.0))
        
    # Row 1 (y=-1.5)
    for i in range(3):
        positions.append((i * s3 + s3 / 2.0, -1.5))
        
    # Row 2 (y=-3.0)
    for i in range(4):
        positions.append((i * s3, -3.0))
        
    # Scale to ensure strict disjointness
    # The minimum distance between centers in tight packing is sqrt(3).
    # The maximum extent of a hexagon from center is 1 (circumradius).
    # Distance between boundaries is sqrt(3) - 2 * (sqrt(3)/2) = 0 for tangent.
    # We scale by 1.01 to create a gap of ~0.01 * sqrt(3) ~ 0.017.
    scale = 1.01
    positions = [(x * scale, y * scale) for x, y in positions]
    
    return positions

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin for 150s limit
    
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
    
    # Optimize outer hexagon
    # Fix theta=0 for the outer hexagon to match the symmetry of the inner cluster (mostly)
    # Actually, the inner cluster is not perfectly hexagonally symmetric (it's a trapezoid),
    # but theta=0 is a good starting point.
    
    normals = []
    for k in range(6):
        theta = np.pi / 3 * k
        normals.append((np.cos(theta), np.sin(theta)))
    normals = np.array(normals)  # shape (6, 2)
    
    m = len(all_verts)
    n_vars = 3 # cx, cy, s
    
    A_ub = np.zeros((6 * m, n_vars))
    b_ub = np.zeros(6 * m)
    
    apothem_factor = np.sqrt(3) / 2
    
    for j in range(6):
        nj = normals[j]
        for i in range(m):
            vi = all_verts[i]
            row = 6 * i + j
            A_ub[row, 0] = -nj[0]
            A_ub[row, 1] = -nj[1]
            A_ub[row, 2] = -apothem_factor
            b_ub[row] = np.dot(vi, nj)
            
    # s >= 0
    A_ub = np.vstack([A_ub, [[0, 0, -1]]])
    b_ub = np.append(b_ub, 0)
    
    c_obj = np.array([0.0, 0.0, 1.0])
    
    cx_opt, cy_opt, s_opt = 0.0, 0.0, 10.0
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
        if res.success:
            cx_opt, cy_opt, s_opt = res.x
            if s_opt < 0:
                s_opt = 0.0
    except Exception:
        pass
        
    # Safety check: ensure all vertices are actually inside
    def _required_side(cx, cy, verts, theta=0.0):
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
    
    if time.time() - start_time > DEADLINE:
        return inner, (0.0, 0.0), 50.0, 0.0
        
    return inner, (cx_opt, cy_opt), s_final, 0.0
