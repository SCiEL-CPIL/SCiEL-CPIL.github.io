import numpy as np
from scipy.optimize import minimize
import time

N = 12

def get_hex_vertices(center, angle_deg, side=1.0):
    """
    Returns the 6 vertices of a regular hexagon.
    center: (x, y)
    angle_deg: rotation in degrees
    side: side length
    """
    cx, cy = center
    angle_rad = np.deg2rad(angle_deg)
    # Vertices at angles 0, 60, 120, 180, 240, 300 relative to the hexagon's local frame
    # Local frame: vertex 0 is at (side, 0)
    angles = np.arange(6) * np.pi / 3.0 + angle_rad
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def polygon_overlap_depth(p1, p2):
    """
    Returns the maximum penetration depth between two convex polygons.
    If disjoint, returns 0.
    If overlapping, returns the minimum distance one polygon must move to separate.
    Uses Separating Axis Theorem (SAT).
    """
    # Axes to test are the normals of the edges of both polygons
    def get_edges_normals(poly):
        n = len(poly)
        normals = []
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            # Normal is perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                normals.append(normal / norm)
        return normals
    
    axes = get_edges_normals(p1) + get_edges_normals(p2)
    
    min_depth = float('inf')
    
    for axis in axes:
        # Project both polygons onto the axis
        proj1 = np.dot(p1, axis)
        proj2 = np.dot(p2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 or max2 < min1:
            return 0.0 # Disjoint
        
        # Calculate overlap depth on this axis
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    return min_depth

def calculate_max_side(params):
    """
    params: array of shape (N, 3) -> [x, y, angle_deg]
    Returns the minimal side length S of the container hexagon (angle 0, center 0,0)
    that contains all inner hexagons.
    """
    max_val = 0.0
    # Container normals for a hexagon with vertex at 0 deg?
    # If vertex at 0, the sides are perpendicular to 30, 90, 150, 210, 270, 330.
    # Normal vectors n_k
    container_angles = np.deg2rad(np.arange(6) * 60 + 30)
    normals = np.column_stack((np.cos(container_angles), np.sin(container_angles)))
    
    # Inradius of container with side S is S * sqrt(3) / 2
    # Distance from center to side is R_cont = S * sqrt(3) / 2
    # So S = 2 * R_cont / sqrt(3)
    
    for i in range(N):
        x, y, ang = params[i]
        verts = get_hex_vertices((x, y), ang)
        # Project all vertices onto all container normals
        # proj = verts @ normals.T
        # We need max(proj) <= R_cont
        # So R_cont >= max( |proj| ) ? No, the container is centered at 0.
        # The constraint is -R_cont <= proj <= R_cont for all vertices and all normals.
        # So R_cont >= max( max(proj), -min(proj) ) = max( |proj| )
        projs = np.dot(verts, normals.T)
        max_proj = np.max(np.abs(projs))
        if max_proj > max_val:
            max_val = max_proj
            
    S = 2.0 * max_val / np.sqrt(3.0)
    return S

def objective(params):
    """
    Objective function for optimization.
    Minimizes S + penalty for overlaps.
    """
    S = calculate_max_side(params)
    
    # Overlap penalty
    penalty = 0.0
    for i in range(N):
        p1 = get_hex_vertices(params[i][:2], params[i][2])
        for j in range(i + 1, N):
            p2 = get_hex_vertices(params[j][:2], params[j][2])
            depth = polygon_overlap_depth(p1, p2)
            if depth > 0:
                penalty += depth ** 2
                
    # If there is overlap, we want to minimize it heavily.
    # Also, we want to minimize S.
    # If penalty is large, it dominates.
    return S + 100.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s margin
    
    # Initial guess: Hexagonal lattice
    # Center at (0,0)
    # Ring 0: (0,0)
    # Ring 1: 6 neighbors
    # Ring 2: Add 5 more to make 12
    
    # Lattice vectors
    a = np.sqrt(3.0)
    e1 = np.array([a, 0])
    e2 = np.array([a/2.0, 1.5]) # Wait, distance is sqrt(3). 
    # Let's use standard axial coords to cartesian
    # q, r -> x = a * (q + r/2), y = a * (sqrt(3)/2 * r) ?? No.
    # Standard hex grid:
    # x = a * (q + r/2)
    # y = a * (sqrt(3)/2 * r)
    # Distance between (0,0) and (1,0) is a.
    # Distance between (0,0) and (0,1) is a.
    
    positions = []
    # Center
    positions.append([0.0, 0.0, 0.0])
    
    # 6 neighbors
    neighbors = [
        [1, 0], [0, 1], [-1, 1], [-1, 0], [0, -1], [1, -1]
    ]
    for q, r in neighbors:
        x = a * (q + r/2.0)
        y = a * (np.sqrt(3.0)/2.0 * r)
        positions.append([x, y, 0.0])
        
    # Need 5 more. Add to the "corners" of the ring 1
    # (2,0), (1,1), (-2,1), (-2,0), (0,-2) ?
    # Let's just pick 5 from the next ring
    next_ring = [
        [2, 0], [1, 1], [0, 2], [-1, 2], [-2, 1], [-2, 0], [-1, -1], [0, -2], [1, -2], [2, -1]
    ]
    # Pick 5 symmetrically?
    # (2,0), (0,2), (-2,1), (-2,0), (0,-2)
    picks = [
        [2, 0], [0, 2], [-2, 1], [-2, 0], [0, -2]
    ]
    for q, r in picks:
        x = a * (q + r/2.0)
        y = a * (np.sqrt(3.0)/2.0 * r)
        positions.append([x, y, 0.0])
        
    # Ensure we have exactly 12
    while len(positions) < N:
        positions.append([0.0, 0.0, 0.0]) # Fallback
    positions = positions[:N]
    
    x0 = np.array(positions)
    
    # Optimization
    best_params = x0
    best_score = objective(x0)
    
    # Use Nelder-Mead
    # Bounds are not strictly defined, but we can assume they stay near origin
    # We will run multiple restarts or just one long run if time permits.
    
    # Define bounds for x,y to prevent drift
    # Max S is likely around 3-4. So x,y should be within [-5, 5]
    bounds = [(-5, 5), (-5, 5), (0, 360)] * N
    
    # Run optimization
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
        
        if res.fun < best_score:
            best_params = res.x
            best_score = res.fun
            
        # Check time
        if time.time() > deadline:
            pass # Just return current best
            
        # If we have time, do a second pass with tighter tolerances or different init
        if time.time() < deadline - 20:
             # Random restarts
             for _ in range(5):
                 if time.time() > deadline - 10: break
                 # Perturb initial guess
                 x0_pert = x0 + np.random.normal(0, 0.5, size=x0.shape)
                 # Keep angles in range
                 x0_pert[:, 2] = np.mod(x0_pert[:, 2], 360)
                 
                 res2 = minimize(objective, x0_pert, method='Nelder-Mead',
                                 options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False})
                 if res2.fun < best_score:
                     best_params = res2.x
                     best_score = res2.fun
                     
    except Exception as e:
        pass # Fallback to initial
    
    # Calculate final S
    S = calculate_max_side(best_params)
    
    # Format output
    inner = []
    for i in range(N):
        x, y, ang = best_params[i]
        ang = np.mod(ang, 360.0)
        inner.append([float(x), float(y), float(ang)])
        
    center = [0.0, 0.0]
    side = float(S)
    angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
