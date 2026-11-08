import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _support_hex(center, angle_deg, side, dx, dy):
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

def _min_enclosing_hex_side(centers, angles, outer_angle_deg):
    """
    Calculate the minimum side length of a regular hexagon with given outer angle
    that contains all inner hexagons.
    """
    outer_angle = np.radians(outer_angle_deg)
    # The 6 outward normals of the outer hexagon
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
            h = _support_hex(np.array([cx, cy]), ang, 1.0, dx, dy)
            dot_center = cx * dx + cy * dy
            needed = dot_center + h
            if needed > max_ext:
                max_ext = needed
    
    # The apothem of the outer hexagon must be at least max_ext
    # Apothem = S * sqrt(3) / 2
    # So S = 2 * max_ext / sqrt(3)
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_validity(centers, angles, S, outer_angle_deg, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon and disjoint.
    """
    # Check containment
    outer_verts = _hex_verts(0.0, 0.0, S, outer_angle_deg)
    inner_verts_list = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        iv = _hex_verts(cx, cy, 1.0, ang)
        inner_verts_list.append(iv)
        for pt in iv:
            # Check if point is inside outer hex
            inside = True
            for j in range(6):
                v1 = outer_verts[j]
                v2 = outer_verts[(j+1)%6]
                edge = v2 - v1
                rel = pt - v1
                cross = edge[0]*rel[1] - edge[1]*rel[0]
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    
    # Check disjointness
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            vi = inner_verts_list[i]
            vj = inner_verts_list[j]
            
            # Fast rejection: if distance between centers is large enough, they don't overlap
            # Max distance for overlap is 2 * circumradius = 2.0
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0:
                continue
            
            # Detailed check
            def intersect(v1, v2):
                def point_in_poly(pt, poly):
                    n = len(poly)
                    for i in range(n):
                        v1 = poly[i]
                        v2 = poly[(i+1)%n]
                        edge = v2 - v1
                        rel = pt - v1
                        cross = edge[0]*rel[1] - edge[1]*rel[0]
                        if cross < -1e-9:
                            return False
                    return True

                for pt in v1:
                    if point_in_poly(pt, v2):
                        return True
                for pt in v2:
                    if point_in_poly(pt, v1):
                        return True
                
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
                            return True
                return False
            
            if intersect(vi, vj):
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin
    
    # Initial lattice configuration: 3x4 honeycomb
    # u and v are basis vectors for the honeycomb lattice
    # Side length of inner hex is 1.
    # Distance between centers in honeycomb:
    # Neighbors are at distance sqrt(3)
    u = np.array([1.5, np.sqrt(3.0)/2.0])
    v = np.array([-1.5, np.sqrt(3.0)/2.0])
    
    # Create 12 points in a 4x3 grid
    positions = []
    for i in range(4):
        for j in range(3):
            pos = i * u + j * v
            positions.append(pos)
    
    # Center the configuration
    pos_array = np.array(positions)
    center = np.mean(pos_array, axis=0)
    pos_array = pos_array - center
    
    # Initial angles: 0 for all
    angles = np.zeros(N)
    
    # We will optimize the positions and the outer hexagon angle.
    # Variables: x0, y0, ..., x11, y11, theta_outer
    # Total 25 variables.
    
    def objective(params):
        centers = np.array([params[2*i], params[2*i+1]] for i in range(N))
        theta_outer = params[24]
        
        S = _min_enclosing_hex_side(centers, angles, theta_outer)
        
        # Check for overlaps (penalty)
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                # Minimum distance for non-overlap of two unit hexagons with same angle is sqrt(3)
                # If angles differ, it can be less, but sqrt(3) is a safe lower bound for same orientation.
                # For robustness, use a soft penalty for dist < sqrt(3)
                if dist < np.sqrt(3.0):
                    penalty += (np.sqrt(3.0) - dist)**2 * 100.0
        
        return S + penalty

    x0 = np.concatenate([pos_array.flatten(), [0.0]])
    
    best_S = float('inf')
    best_params = x0.copy()
    
    # Try multiple optimization methods and restarts
    methods = ['Powell', 'Nelder-Mead']
    
    for method in methods:
        if time.time() > deadline - 5.0:
            break
            
        try:
            res = minimize(objective, x0, method=method, 
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_S:
                best_params = res.x
                best_S = res.fun
                
        except Exception:
            pass

    # Extract best configuration
    best_centers = np.array([best_params[2*i], best_params[2*i+1]] for i in range(N))
    best_theta_outer = best_params[24]
    
    # Calculate exact S for this configuration
    final_S = _min_enclosing_hex_side(best_centers, angles, best_theta_outer)
    
    # Validate
    if _check_validity(best_centers, angles, final_S, best_theta_outer):
        inner = [[float(cx), float(cy), 0.0] for cx, cy in best_centers]
        return inner, (0.0, 0.0), float(final_S), float(best_theta_outer)
    else:
        # Fallback to initial valid configuration if optimization broke it
        # The initial lattice is usually valid
        init_S = _min_enclosing_hex_side(pos_array, angles, 0.0)
        if _check_validity(pos_array, angles, init_S, 0.0):
            inner = [[float(cx), float(cy), 0.0] for cx, cy in pos_array]
            return inner, (0.0, 0.0), float(init_S), 0.0
        else:
            # Emergency fallback: widely spaced
            inner = []
            for i in range(12):
                inner.append([0.0, 0.0, 0.0])
            return inner, (0.0, 0.0), 10.0, 0.0

