import numpy as np
from scipy.optimize import minimize
import time

N = 12

def _hex_vertices(center, angle):
    """
    Returns vertices of a regular hexagon with side length 1.
    center: (x, y)
    angle: rotation in radians
    """
    cx, cy = center
    # Vertices of unit hexagon (side 1) centered at origin, angle 0
    # Vertex 0 is at (1, 0)
    angles = np.deg2rad(np.arange(0, 360, 60) + np.rad2deg(angle))
    # Actually, simpler:
    base_angles = np.deg2rad(np.arange(0, 360, 60))
    rotated_angles = base_angles + angle
    verts = np.column_stack((np.cos(rotated_angles), np.sin(rotated_angles)))
    verts += center
    return verts

def _sat_overlap_penalty(v1, v2):
    """
    Computes a smooth penalty for overlap between two convex polygons.
    v1, v2: arrays of vertices (N, 2)
    Returns 0 if separated, positive value if overlapping.
    """
    penalty = 0.0
    # Get normals for v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        # Project v1
        proj1 = np.dot(v1, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        # Project v2
        proj2 = np.dot(v2, normal)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check separation
        if max1 < min2 or max2 < min1:
            return 0.0 # Separating axis found
        
        # Calculate overlap depth
        # The overlap in this axis is the distance between the inner boundaries
        # min(max1, max2) - max(min1, min2)
        # If this is negative, they overlap.
        overlap = max(min1, min2) - min(max1, max2)
        if overlap < 0:
            penalty += overlap * overlap
            
    # Check normals for v2 as well (though for convex polygons, checking all edges of both is sufficient)
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        normal = normal / np.linalg.norm(normal)
        
        proj1 = np.dot(v1, normal)
        min1, max1 = np.min(proj1), np.max(proj1)
        
        proj2 = np.dot(v2, normal)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
            
        overlap = max(min1, min2) - min(max1, max2)
        if overlap < 0:
            penalty += overlap * overlap
            
    return penalty

def _objective(params):
    """
    params: array of length 3*N: [x1, y1, a1, x2, y2, a2, ...]
    Returns: S_min + lambda * sum(overlap_penalties)
    """
    centers = params[0::3]
    angles = params[1::3]
    
    # Compute vertices for all hexagons
    verts_list = []
    for i in range(N):
        c = (centers[2*i], centers[2*i+1])
        a = angles[i]
        verts_list.append(_hex_vertices(c, a))
        
    # Compute S_min
    # Normals of outer hexagon (side S centered at origin)
    # The distance from center to edge is h = S * sqrt(3) / 2
    # Condition: n . v <= h  =>  S >= 2/sqrt(3) * max(n . v)
    # We want to minimize S, so S_min = 2/sqrt(3) * max(n . v)
    
    # Outer hexagon normals (angles 0, 60, 120, 180, 240, 300)
    outer_normals = np.array([
        [np.cos(0), np.sin(0)],
        [np.cos(np.pi/3), np.sin(np.pi/3)],
        [np.cos(2*np.pi/3), np.sin(2*np.pi/3)],
        [np.cos(np.pi), np.sin(np.pi)],
        [np.cos(4*np.pi/3), np.sin(4*np.pi/3)],
        [np.cos(5*np.pi/3), np.sin(5*np.pi/3)]
    ])
    
    max_proj = -np.inf
    for v in verts_list:
        # v is (6, 2)
        # project onto all 6 normals
        projs = np.dot(v, outer_normals.T) # (6, 6)
        max_proj = max(max_proj, np.max(projs))
        
    S_min = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Compute overlap penalty
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            penalty += _sat_overlap_penalty(verts_list[i], verts_list[j])
            
    lambda_weight = 100.0
    return S_min + lambda_weight * penalty

def construct():
    start_time = time.time()
    deadline = 120.0 # 120 seconds limit, use 110 to be safe
    
    # Initial positions: Hexagonal lattice
    a1 = np.array([np.sqrt(3), 0])
    a2 = np.array([np.sqrt(3)/2, 1.5])
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            p = i*a1 + j*a2
            pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    init_pts = pts[idx]
    
    # Initial angles: 0
    init_angles = np.zeros(N)
    
    # Flatten parameters
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = init_pts[i][0]
        x0[3*i+1] = init_pts[i][1]
        x0[3*i+2] = init_angles[i]
        
    best_params = x0.copy()
    best_score = _objective(x0)
    
    # Optimization
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8})
        
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x
            
        # Check time
        elapsed = time.time() - start_time
        if elapsed < deadline:
            # Maybe try a second optimization with different start or more iterations?
            # Or just return the best found.
            pass
            
    except Exception as e:
        pass # Keep best_params
        
    # Extract best parameters
    centers = best_params[0::3]
    angles = best_params[1::3]
    
    # Calculate final S_min for the best configuration
    # Re-use logic from _objective but without penalty
    verts_list = []
    for i in range(N):
        c = (centers[2*i], centers[2*i+1])
        a = angles[i]
        verts_list.append(_hex_vertices(c, a))
        
    outer_normals = np.array([
        [np.cos(0), np.sin(0)],
        [np.cos(np.pi/3), np.sin(np.pi/3)],
        [np.cos(2*np.pi/3), np.sin(2*np.pi/3)],
        [np.cos(np.pi), np.sin(np.pi)],
        [np.cos(4*np.pi/3), np.sin(4*np.pi/3)],
        [np.cos(5*np.pi/3), np.sin(5*np.pi/3)]
    ])
    
    max_proj = -np.inf
    for v in verts_list:
        projs = np.dot(v, outer_normals.T)
        max_proj = max(max_proj, np.max(projs))
        
    S_final = (2.0 / np.sqrt(3.0)) * max_proj
    
    # Construct return values
    inner = []
    for i in range(N):
        x = centers[2*i]
        y = centers[2*i+1]
        angle_deg = np.rad2deg(angles[i]) % 360.0
        inner.append([x, y, angle_deg])
        
    center = (0.0, 0.0)
    side = S_final
    angle = 0.0
    
    return inner, center, side, angle

