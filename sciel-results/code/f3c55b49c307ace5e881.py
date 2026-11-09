# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s, tol=1e-6, edge_tol=1e-8):
    """Check validity of a packing configuration."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons have unit edges
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > edge_tol:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            normals = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n = n / n_norm
                        normals.append(n)
            
            separated = False
            for n in normals:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap > -tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _compute_min_s(inner, outer_angle_deg):
    """Compute the minimum outer side length needed to contain all inner hexagons.
    Assumes outer center is (0,0)."""
    oang = np.deg2rad(outer_angle_deg)
    
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # n_k angles in radians
    n_angles = oang + np.deg2rad(30.0) + np.deg2rad(60.0 * np.arange(6))
    normals = np.column_stack([np.cos(n_angles), np.sin(n_angles)])
    
    dots = all_verts @ normals.T  # shape (72, 6)
    max_dots = dots.max(axis=0)  # shape (6,)
    s_min = (2.0 / math.sqrt(3.0)) * max_dots.max()
    
    return s_min

def _objective(params):
    """Objective: minimize s + penalty for overlaps.
    Params: 24 center coords + 12 angles + 1 outer angle = 37 params."""
    cx_cy = params[:24].reshape(12, 2)
    inner_angles = params[24:36]
    outer_angle = params[36]
    
    inner = np.zeros((12, 3))
    inner[:, 0] = cx_cy[:, 0]
    inner[:, 1] = cx_cy[:, 1]
    inner[:, 2] = inner_angles
    
    s = _compute_min_s(inner, outer_angle)
    
    # Penalty for overlaps: simple distance check
    # If centers are too close, it's likely an overlap.
    # Distance between centers of touching hexes is sqrt(3).
    penalty = 0.0
    for i in range(12):
        for j in range(i+1, 12):
            dx = cx_cy[i, 0] - cx_cy[j, 0]
            dy = cx_cy[i, 1] - cx_cy[j, 1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < math.sqrt(3) - 1e-6:
                penalty += (math.sqrt(3) - dist)**2
    
    return s + 10.0 * penalty

def hexagon_packing_12():
    start_time = time.time()
    deadline = 620.0
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    # Initial configuration: 3-4-3-2 hexagonal grid
    # Hex lattice spacing: dx = sqrt(3), dy = 1.5
    # Let's define positions relative to center (0,0)
    # We want to center the cluster.
    # Rows:
    # y = -4.5: 2 hexes (x = -0.5*sqrt(3), 0.5*sqrt(3))
    # y = -3.0: 3 hexes (x = -sqrt(3), 0, sqrt(3))
    # y = -1.5: 4 hexes (x = -1.5*sqrt(3), -0.5*sqrt(3), 0.5*sqrt(3), 1.5*sqrt(3))
    # y = 0.0: 3 hexes (x = -sqrt(3), 0, sqrt(3))
    # Total 12.
    # Wait, 2+3+4+3 = 12.
    # Let's shift y so the center of mass is 0.
    # y values: -4.5, -3.0, -1.5, 0.0.
    # Sum y = 2*(-4.5) + 3*(-3.0) + 4*(-1.5) + 3*(0.0) = -9 - 9 - 6 = -24.
    # Avg y = -24/12 = -2.0.
    # Shift all y by +2.0.
    # New y: -2.5, -1.0, 0.5, 2.0.
    
    sqrt3 = math.sqrt(3)
    positions = []
    # Row 0 (y=-2.5): 2 hexes
    positions.append((-0.5*sqrt3, -2.5))
    positions.append((0.5*sqrt3, -2.5))
    # Row 1 (y=-1.0): 3 hexes
    positions.append((-sqrt3, -1.0))
    positions.append((0.0, -1.0))
    positions.append((sqrt3, -1.0))
    # Row 2 (y=0.5): 4 hexes
    positions.append((-1.5*sqrt3, 0.5))
    positions.append((-0.5*sqrt3, 0.5))
    positions.append((0.5*sqrt3, 0.5))
    positions.append((1.5*sqrt3, 0.5))
    # Row 3 (y=2.0): 3 hexes
    positions.append((-sqrt3, 2.0))
    positions.append((0.0, 2.0))
    positions.append((sqrt3, 2.0))
    
    initial_params = np.zeros(37)
    for i in range(12):
        initial_params[2*i] = positions[i][0]
        initial_params[2*i+1] = positions[i][1]
    # Angles default to 0
    # Outer angle default to 0
    
    # Bounds
    bounds = [(-10, 10)] * 24
    bounds += [(-180, 180)] * 12
    bounds += [(-30, 30)]
    
    # Optimization
    try:
        # Use Powell for robustness
        res = minimize(
            _objective,
            initial_params,
            method='Powell',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-8}
        )
        
        if res.fun < best_s:
            cx_cy = res.x[:24].reshape(12, 2)
            inner_angles = res.x[24:36]
            outer_angle = res.x[36]
            inner = np.zeros((12, 3))
            inner[:, 0] = cx_cy[:, 0]
            inner[:, 1] = cx_cy[:, 1]
            inner[:, 2] = inner_angles
            s = _compute_min_s(inner, outer_angle)
            outer = np.array([0.0, 0.0, outer_angle])
            if _is_valid(inner, outer, s):
                best_s = s
                best_inner = inner.copy()
                best_outer = outer.copy()
                
    except Exception as e:
        print(f"Powell failed: {e}")
        
    # If still no valid solution or we have time, try Nelder-Mead refinement
    if best_inner is not None and time.time() < deadline - 10:
        best_x = np.concatenate([best_inner[:, :2].flatten(), best_inner[:, 2], [best_outer[2]]])
        try:
            res2 = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            if res2.fun < best_s:
                cx_cy = res2.x[:24].reshape(12, 2)
                inner_angles = res2.x[24:36]
                outer_angle = res2.x[36]
                inner = np.zeros((12, 3))
                inner[:, 0] = cx_cy[:, 0]
                inner[:, 1] = cx_cy[:, 1]
                inner[:, 2] = inner_angles
                s = _compute_min_s(inner, outer_angle)
                outer = np.array([0.0, 0.0, outer_angle])
                if _is_valid(inner, outer, s):
                    best_s = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
        except Exception as e:
            print(f"Nelder-Mead failed: {e}")
            
    # Fallback
    if best_inner is None:
        # Use the initial grid as fallback if it's valid
        inner = np.zeros((12, 3))
        for i in range(12):
            inner[i, 0] = positions[i][0]
            inner[i, 1] = positions[i][1]
            inner[i, 2] = 0.0
        s = _compute_min_s(inner, 0.0)
        outer = np.array([0.0, 0.0, 0.0])
        if _is_valid(inner, outer, s):
            return inner, outer, s
        else:
            # Very loose fallback
            return np.zeros((12,3)), np.array([0,0,0]), 10.0

    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
