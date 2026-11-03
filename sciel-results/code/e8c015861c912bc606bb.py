import numpy as np
import time
from scipy.optimize import minimize

N = 12

def _hex_vertices(cx, cy, angle_deg, scale=1.0):
    """
    Returns vertices of a regular hexagon.
    angle_deg: Rotation in degrees.
    scale: Side length.
    Vertices are ordered CCW.
    """
    # Flat-top hexagon vertices relative to center:
    # (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    # This corresponds to angle 0 pointing to the right.
    angles = np.deg2rad([0, 60, 120, 180, 240, 300]) + np.deg2rad(angle_deg)
    x = cx + scale * np.cos(angles)
    y = cy + scale * np.sin(angles)
    return np.column_stack((x, y))

def _sat_overlap_depth(p1, p2):
    """
    Returns the minimum separation distance between two convex polygons.
    Positive if separated, negative if overlapping (depth of penetration).
    """
    # Get edges and normals
    n1 = len(p1)
    n2 = len(p2)
    
    min_sep = float('inf')
    
    # Check axes from p1
    for i in range(n1):
        # Edge vector
        v = p1[(i+1)%n1] - p1[i]
        # Normal (perpendicular)
        # For CCW polygon, normal points outward: (dy, -dx)
        nx = v[1]
        ny = -v[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-9: continue
        nx /= norm
        ny /= norm
        
        # Project p1
        proj1 = np.dot(p1, [nx, ny])
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        # Project p2
        proj2 = np.dot(p2, [nx, ny])
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        # Separation
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            sep = 0 # Overlapping on this axis
            
        if sep < min_sep:
            min_sep = sep
            
    # Check axes from p2
    for i in range(n2):
        v = p2[(i+1)%n2] - p2[i]
        nx = v[1]
        ny = -v[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-9: continue
        nx /= norm
        ny /= norm
        
        proj1 = np.dot(p1, [nx, ny])
        min1 = np.min(proj1)
        max1 = np.max(proj1)
        
        proj2 = np.dot(p2, [nx, ny])
        min2 = np.min(proj2)
        max2 = np.max(proj2)
        
        if max1 < min2:
            sep = min2 - max1
        elif max2 < min1:
            sep = min1 - max2
        else:
            sep = 0
            
        if sep < min_sep:
            min_sep = sep
            
    return min_sep

def _point_to_polygon_distance(p, poly):
    """
    Returns distance from point p to polygon poly.
    0 if inside.
    Positive if outside.
    Assumes poly is CCW.
    """
    # Check if inside
    n = len(poly)
    inside = True
    min_dist = float('inf')
    
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1)%n]
        # Edge vector
        e = p2 - p1
        # Normal pointing inward?
        # For CCW, outward normal is (dy, -dx).
        # Inward normal is (-dy, dx).
        nx = -e[1]
        ny = e[0]
        norm = np.linalg.norm([nx, ny])
        if norm < 1e-9: continue
        nx /= norm
        ny /= norm
        
        # Distance from p to line containing edge
        # Vector from p1 to p
        v = p - p1
        dist = np.dot(v, [nx, ny])
        
        # If dist < 0, point is outside this edge
        if dist < -1e-9:
            inside = False
            # Distance to the edge segment?
            # For convex polygon, if outside, distance is to the closest edge or vertex.
            # We can just use the perpendicular distance if the projection falls within the edge,
            # else distance to vertex.
            # Simplified: Just use perpendicular distance as a lower bound/estimate?
            # Better: Calculate distance to segment p1-p2.
            t = np.dot(v, e) / (np.dot(e, e))
            t = np.clip(t, 0, 1)
            proj = p1 + t * e
            d = np.linalg.norm(p - proj)
            if d < min_dist:
                min_dist = d
            
    if inside:
        return 0.0
    else:
        return min_dist

def objective(x):
    """
    x: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, L]
    """
    coords = x[:24].reshape(12, 2)
    angles = x[24:36]
    L = x[36]
    
    if L < 0:
        return 1e6
        
    # Outer hexagon
    outer_verts = _hex_vertices(0, 0, 0, L)
    
    # Inner hexagons
    inner_verts = []
    for i in range(12):
        v = _hex_vertices(coords[i][0], coords[i][1], angles[i], 1.0)
        inner_verts.append(v)
        
    penalty = 0.0
    
    # Overlap penalty
    for i in range(12):
        for j in range(i+1, 12):
            sep = _sat_overlap_depth(inner_verts[i], inner_verts[j])
            if sep < 0:
                penalty += (-sep)**2
                
    # Outside penalty
    for i in range(12):
        # Check vertices of inner hex i against outer hex
        for v in inner_verts[i]:
            d = _point_to_polygon_distance(v, outer_verts)
            if d > 0:
                penalty += d**2
                
    return L + 100.0 * penalty

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_x = None
    
    # Initial Guess 1: Honeycomb
    # Center
    # 6 neighbors
    # 5 in bays
    
    x0 = np.zeros(37)
    
    # Hex 0
    x0[0] = 0.0
    x0[1] = 0.0
    x0[2] = 0.0
    
    # Hex 1-6
    for k in range(6):
        angle = k * 60
        rad = np.deg2rad(angle)
        x0[3 + k*3] = 2.0 * np.cos(rad)
        x0[4 + k*3] = 2.0 * np.sin(rad)
        x0[5 + k*3] = 0.0
        
    # Hex 7-11 (5 bays)
    bay_angles = [30, 90, 150, 210, 270]
    for k in range(5):
        angle = bay_angles[k]
        rad = np.deg2rad(angle)
        dist = 3.5 # Guess
        idx = 7 + k
        x0[idx*3] = dist * np.cos(rad)
        x0[idx*3+1] = dist * np.sin(rad)
        x0[idx*3+2] = 0.0
        
    x0[36] = 5.0 # Initial L
    
    initial_guesses = [x0]
    
    # Initial Guess 2: Random
    for _ in range(5):
        xr = np.random.rand(24) * 6 - 3
        ar = np.random.rand(12) * 360
        Lr = 6.0
        x_r = np.concatenate([xr, ar, [Lr]])
        initial_guesses.append(x_r)
        
    for x_init in initial_guesses:
        if time.time() - start_time > 110.0:
            break
            
        try:
            res = minimize(objective, x_init, method='Powell', options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
            
            if res.fun < best_score:
                best_score = res.fun
                best_x = res.x
                
                # Check validity
                coords = best_x[:24].reshape(12, 2)
                angles = best_x[24:36]
                L = best_x[36]
                
                # Quick check if penalty is low enough
                # Re-evaluate penalty
                outer_verts = _hex_vertices(0, 0, 0, L)
                inner_verts = [_hex_vertices(coords[i][0], coords[i][1], angles[i], 1.0) for i in range(12)]
                
                p = 0.0
                for i in range(12):
                    for j in range(i+1, 12):
                        sep = _sat_overlap_depth(inner_verts[i], inner_verts[j])
                        if sep < 0: p += (-sep)**2
                for i in range(12):
                    for v in inner_verts[i]:
                        d = _point_to_polygon_distance(v, outer_verts)
                        if d > 0: p += d**2
                        
                if p < 1e-4: # Valid
                    inner = [[coords[i][0], coords[i][1], angles[i]] for i in range(12)]
                    return inner, (0.0, 0.0), L, 0.0
                    
        except Exception:
            pass
            
    # If no valid solution found, return best effort or fallback
    if best_x is not None:
        coords = best_x[:24].reshape(12, 2)
        angles = best_x[24:36]
        L = best_x[36]
        inner = [[coords[i][0], coords[i][1], angles[i]] for i in range(12)]
        return inner, (0.0, 0.0), L, 0.0
        
    # Fallback
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
