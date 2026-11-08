"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses a structured search over symmetric configurations with local refinement.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_hex(cx, cy, angle_deg, side, dx, dy):
    """Support function of a regular hexagon in direction (dx, dy)."""
    d = np.array([dx, dy])
    dn = np.linalg.norm(d)
    if dn < 1e-12:
        return 0.0
    d = d / dn
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    vs = side * np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ d
    return np.max(dots)

def _min_enclosing_hex_side(centers, angles, outer_angle_deg=0.0):
    """
    Calculate the minimum side length of a regular hexagon with given outer angle
    that contains all inner hexagons.
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    max_ext = 0.0
    for k in range(len(centers)):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx = normals[ni, 0]
            dy = normals[ni, 1]
            h = _support_hex(cx, cy, ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            needed = dot_center + h
            if needed > max_ext:
                max_ext = needed
    
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_validity(centers, angles, S, outer_angle_deg=0.0, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon and disjoint.
    """
    # Check containment using support function (faster)
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    apothem = S * np.sqrt(3.0) / 2.0
    
    for k in range(len(centers)):
        cx, cy = centers[k]
        ang = angles[k]
        for ni in range(6):
            dx = normals[ni, 0]
            dy = normals[ni, 1]
            h = _support_hex(cx, cy, ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            if dot_center + h > apothem + tol:
                return False
    
    # Check disjointness using vertex-edge intersection
    inner_verts_list = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(iv)
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            
            # Check if any vertex of one is inside the other
            for pt in vi:
                inside = True
                for k in range(6):
                    v1 = vj[k]
                    v2 = vj[(k+1)%6]
                    edge = v2 - v1
                    rel = pt - v1
                    cross = edge[0]*rel[1] - edge[1]*rel[0]
                    if cross < -1e-9:
                        inside = False
                        break
                if inside:
                    return False
            
            for pt in vj:
                inside = True
                for k in range(6):
                    v1 = vi[k]
                    v2 = vi[(k+1)%6]
                    edge = v2 - v1
                    rel = pt - v1
                    cross = edge[0]*rel[1] - edge[1]*rel[0]
                    if cross < -1e-9:
                        inside = False
                        break
                if inside:
                    return False
            
            # Check edge-edge intersections
            def segments_intersect(p1, p2, p3, p4):
                def cross(o, a, b):
                    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                
                d1 = cross(p3, p4, p1)
                d2 = cross(p3, p4, p2)
                d3 = cross(p1, p2, p3)
                d4 = cross(p1, p2, p4)
                
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
                return False
            
            for i in range(6):
                p1 = vi[i]
                p2 = vi[(i+1)%6]
                for j in range(6):
                    p3 = vj[j]
                    p4 = vj[(j+1)%6]
                    if segments_intersect(p1, p2, p3, p4):
                        return False
                
    return True

def _overlap_penalty(centers, angles):
    """
    Compute a smooth penalty for overlapping hexagons.
    Uses minimum distance between hexagons based on support functions.
    """
    penalty = 0.0
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            ci = centers[i]
            cj = centers[j]
            ai = angles[i]
            aj = angles[j]
            
            # Sample multiple directions to estimate minimum distance
            min_dist = float('inf')
            for theta_deg in range(0, 360, 15):
                theta = np.radians(theta_deg)
                dx = np.cos(theta)
                dy = np.sin(theta)
                
                hi = _support_hex(ci[0], ci[1], ai, 1.0, dx, dy)
                hj = _support_hex(cj[0], cj[1], aj, 1.0, dx, dy)
                
                # Distance in this direction
                dist_dir = abs(ci[0]*dx + ci[1]*dy - cj[0]*dx - cj[1]*dy) - hi - hj
                if dist_dir < min_dist:
                    min_dist = dist_dir
            
            if min_dist < 0:
                penalty += min_dist * min_dist * 100.0
            elif min_dist < 0.1:
                penalty += min_dist * min_dist * 10.0
    
    return penalty

def _objective(params, outer_angle_deg=0.0):
    """
    Objective function for optimization.
    params: 3*N array [x0, y0, a0, x1, y1, a1, ...]
    """
    centers = np.zeros((N, 2))
    angles = np.zeros(N)
    for i in range(N):
        centers[i, 0] = params[3*i]
        centers[i, 1] = params[3*i+1]
        angles[i] = params[3*i+2]
    
    S = _min_enclosing_hex_side(centers, angles, outer_angle_deg)
    penalty = _overlap_penalty(centers, angles)
    
    return S + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_params = None
    best_score = float('inf')
    best_inner = None
    
    # Strategy 1: Hexagonal lattice with various scales and rotations
    # For 12 hexagons, a 3x4 or 2x6 grid or other arrangements
    
    # Lattice vectors for flat-top hexagons
    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    
    # Try different grid configurations
    grid_configs = [
        # (rows, cols, scale, base_angle)
        (3, 4, 1.0, 0.0),
        (3, 4, 1.0, 30.0),
        (4, 3, 1.0, 0.0),
        (4, 3, 1.0, 30.0),
        (2, 6, 1.0, 0.0),
        (6, 2, 1.0, 0.0),
    ]
    
    for rows, cols, scale, base_angle in grid_configs:
        if time.time() > deadline:
            break
        
        inner = []
        for i in range(rows):
            for j in range(cols):
                pos = scale * (i * u + j * v)
                inner.append([float(pos[0]), float(pos[1]), float(base_angle)])
        
        # Center the configuration
        if len(inner) > 0:
            cx_mean = np.mean([p[0] for p in inner])
            cy_mean = np.mean([p[1] for p in inner])
            inner = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner]
        
        centers = np.array([[p[0], p[1]] for p in inner])
        angles = np.array([p[2] for p in inner])
        
        S_val = _min_enclosing_hex_side(centers, angles, 0.0)
        
        if _check_validity(centers, angles, S_val, 0.0):
            if S_val < best_score:
                best_score = S_val
                best_params = [p for row in inner for p in row]
                best_inner = inner
    
    # Strategy 2: Try with outer hexagon rotated
    for outer_angle in [0.0, 15.0, 30.0]:
        if time.time() > deadline:
            break
        
        # Re-evaluate best with different outer angle
        if best_params is not None:
            centers = np.array([[best_params[3*i], best_params[3*i+1]] for i in range(N)])
            angles = np.array([best_params[3*i+2] for i in range(N)])
            
            S_val = _min_enclosing_hex_side(centers, angles, outer_angle)
            
            if _check_validity(centers, angles, S_val, outer_angle):
                if S_val < best_score:
                    best_score = S_val
                    best_inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    
    # Strategy 3: Local refinement
    if best_params is not None and time.time() < deadline - 10:
        # Try refinement with different outer angles
        for outer_angle in [0.0, 15.0]:
            if time.time() > deadline - 5:
                break
            
            result = minimize(
                lambda p: _objective(p, outer_angle),
                np.array(best_params),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-8}
            )
            
            if result.fun < best_score:
                params = result.x
                centers = np.array([[params[3*i], params[3*i+1]] for i in range(N)])
                angles = np.array([params[3*i+2] for i in range(N)])
                S_val = _min_enclosing_hex_side(centers, angles, outer_angle)
                
                if _check_validity(centers, angles, S_val, outer_angle):
                    if S_val < best_score:
                        best_score = S_val
                        best_inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    
    # If no valid configuration found, create a guaranteed valid one
    if best_inner is None:
        # Use a simple grid with extra spacing
        scale = 1.2
        inner = []
        for i in range(3):
            for j in range(4):
                pos = scale * (i * u + j * v)
                inner.append([float(pos[0]), float(pos[1]), 0.0])
        
        cx_mean = np.mean([p[0] for p in inner])
        cy_mean = np.mean([p[1] for p in inner])
        inner = [[p[0]-cx_mean, p[1]-cy_mean, p[2]] for p in inner]
        
        centers = np.array([[p[0], p[1]] for p in inner])
        angles = np.array([0.0]*N)
        S_val = _min_enclosing_hex_side(centers, angles, 0.0)
        
        return inner, (0.0, 0.0), float(S_val), 0.0
    
    # Return the best configuration
    centers = np.array([[p[0], p[1]] for p in best_inner])
    angles = np.array([p[2] for p in best_inner])
    
    # Try to find the best outer angle
    best_S = float('inf')
    best_outer_angle = 0.0
    for outer_angle in [0.0, 15.0, 30.0]:
        S_val = _min_enclosing_hex_side(centers, angles, outer_angle)
        if _check_validity(centers, angles, S_val, outer_angle):
            if S_val < best_S:
                best_S = S_val
                best_outer_angle = outer_angle
    
    return best_inner, (0.0, 0.0), float(best_S), float(best_outer_angle)
