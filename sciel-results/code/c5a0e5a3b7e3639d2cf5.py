import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(center, side, angle_deg):
    """Generate vertices of a regular hexagon."""
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    # Vertices of a regular hexagon of side 1 centered at origin
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack((xs, ys))

def point_in_polygon(point, polygon):
    """Check if point is inside polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xint = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x != p2x:
                        if x <= xint:
                            inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def polygons_overlap(poly1, poly2):
    """Check if two convex polygons overlap.
    Returns True if they overlap.
    For convex polygons of same size, checking vertices is sufficient 
    unless they are perfectly nested (impossible for same size distinct centers) 
    or edges cross without vertex containment (impossible for convex if no vertex is inside).
    Actually, if two convex polygons overlap, at least one vertex of one must be inside the other.
    """
    for p in poly1:
        if point_in_polygon(p, poly2):
            return True
    for p in poly2:
        if point_in_polygon(p, poly1):
            return True
    return False

def compute_overlap_penalty(polygons):
    """Compute a smooth penalty for overlapping polygons.
    We use the minimum distance between polygons if they don't overlap, 
    or a large penalty if they do. 
    For optimization, a hard constraint is hard. 
    We will check for overlap. If overlap, we calculate the 'depth' roughly 
    by the distance between centers if they are close, or just a large constant.
    A better metric: For each pair, if they overlap, add penalty proportional to 
    the amount of overlap. 
    Simple approach: If overlap, return large number. 
    To make it differentiable, we can't easily. 
    We will use a penalty based on the minimum distance between vertices of different polygons.
    If min_dist < 0 (overlap), penalty is large.
    """
    penalty = 0.0
    n_polys = len(polygons)
    for i in range(n_polys):
        for j in range(i + 1, n_polys):
            # Check overlap
            if polygons_overlap(polygons[i], polygons[j]):
                # Estimate depth: 
                # Find closest pair of points? Too slow.
                # Just use center distance if < 2.0 (touching distance for aligned hexes is ~1.732 to 2.0)
                c1 = np.mean(polygons[i], axis=0)
                c2 = np.mean(polygons[j], axis=0)
                dist = np.linalg.norm(c1 - c2)
                # Minimum distance between centers for non-overlap is roughly 1.732 (apothem*2) to 2.0 (radius*2)
                # If dist < 1.732, they definitely overlap.
                # We add a penalty proportional to (1.732 - dist) if dist < 1.732, else large constant if overlapping.
                # Actually, if they overlap, the vertex check failed. 
                # Let's just add a large penalty.
                penalty += 1000.0
                # To help gradient-free methods, we can add a term based on center distance
                if dist < 2.0:
                    penalty += (2.0 - dist) ** 2 * 10.0
    return penalty

def construct_impl():
    start_time = time.time()
    deadline = 145.0  # 5s margin from 150s limit

    # Initial configuration: Hexagonal lattice
    # 1 center, 6 ring 1, 4 ring 2
    # Ring 1: distance 2 (touching)
    # Ring 2: distance 2*sqrt(3) ~ 3.464, placed in gaps?
    # Let's place Ring 2 at angles 30, 150, 270, 390 (which is 30) -> 30, 150, 270, 30? 
    # Symmetric choice: 30, 150, 270, 390 is not symmetric.
    # Let's use 30, 90, 150, 210? No.
    # Let's use 30, 150, 270, 390 (30) -> 30, 150, 270, 30. 
    # Better: 30, 150, 270, 390 is 4 points. 
    # Let's try 30, 150, 270, 390 (mod 360 = 30). 
    # Angles: 30, 150, 270, 30. Two at 30? No.
    # Let's use 30, 150, 270, 390 -> 30, 150, 270, 30. 
    # Okay, let's just pick 4 symmetric positions.
    # 30, 150, 270, 390 is not 4 distinct.
    # 30, 150, 270, 390 mod 360 = 30, 150, 270, 30.
    # Let's use 30, 150, 270, 30. 
    # Actually, let's use 30, 90, 150, 210? 
    # Let's just use the first 4 positions from the 6 possible gap positions.
    # Gap positions are at 30, 90, 150, 210, 270, 330.
    # We pick 30, 90, 150, 210.
    
    inner_init = []
    # Center
    inner_init.append([0.0, 0.0, 0.0])
    # Ring 1
    for i in range(6):
        angle = i * 60
        rad = np.deg2rad(angle)
        x = 2.0 * np.cos(rad)
        y = 2.0 * np.sin(rad)
        inner_init.append([x, y, 0.0])
    # Ring 2 (4 hexagons)
    # Placed at distance 2*sqrt(3) at angles 30, 90, 150, 210
    for i in range(4):
        angle = 30 + i * 60 # 30, 90, 150, 210
        rad = np.deg2rad(angle)
        x = 2 * np.sqrt(3) * np.cos(rad)
        y = 2 * np.sqrt(3) * np.sin(rad)
        inner_init.append([x, y, 0.0])
    
    # Parameters: x, y, angle for each inner hex, plus outer angle
    # Total 11 * 3 + 1 = 34 variables
    x0 = []
    for h in inner_init:
        x0.append(h[0])
        x0.append(h[1])
        x0.append(h[2])
    x0.append(0.0) # outer angle
    
    def objective(params):
        # Unpack
        inners = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inners.append([x, y, a])
        outer_angle = params[N*3]
        
        # Generate polygons
        polygons = []
        for h in inners:
            verts = hexagon_vertices((h[0], h[1]), 1.0, h[2])
            polygons.append(verts)
            
        # Overlap penalty
        pen_overlap = compute_overlap_penalty(polygons)
        
        # Outer hexagon constraint
        # Minimize side length L. 
        # Equivalent to minimizing apothem A = L * sqrt(3) / 2.
        # So we minimize A.
        # Constraint: All vertices must satisfy n_i . p <= A for all 6 normals.
        # Normals are at angles outer_angle, outer_angle+60, ...
        # n_i = (cos(theta_i), sin(theta_i))
        # We need A >= max( n_i . p ) for all i, p.
        # So A_min = max( n_i . p ).
        # We return A_min + pen_overlap.
        
        max_proj = 0.0
        for i in range(6):
            theta = np.deg2rad(outer_angle + i * 60)
            nx = np.cos(theta)
            ny = np.sin(theta)
            for poly in polygons:
                # Project all vertices
                projections = poly[:, 0] * nx + poly[:, 1] * ny
                max_p = np.max(projections)
                if max_p > max_proj:
                    max_proj = max_p
        
        # A_min is max_proj.
        # L = A_min * 2 / sqrt(3)
        # We want to minimize L, so we minimize A_min.
        # Return A_min + penalty.
        
        return max_proj + pen_overlap

    # Optimize
    # Nelder-Mead is good for non-differentiable penalties.
    # Powell is also good.
    # Let's try Nelder-Mead with reasonable settings.
    
    best_result = None
    best_val = float('inf')
    
    # Multiple restarts if time permits
    # For now, one run with high maxiter.
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
    
    if result.fun < best_val:
        best_val = result.fun
        best_result = result.x
        
    # Check time
    if time.time() < deadline - 10:
        # Try a second run with different initial guess?
        # Or just refine?
        # Let's try Powell for refinement
        result2 = minimize(objective, best_result, method='Powell', 
                           options={'maxiter': 100, 'ftol': 1e-8})
        if result2.fun < best_val:
            best_val = result2.fun
            best_result = result2.x

    # Extract final solution
    best_params = best_result
    inner_final = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner_final.append([x, y, a])
    
    outer_angle_final = best_params[N*3]
    
    # Calculate final side length L
    # Recompute A_min for the final configuration to be sure
    polygons = []
    for h in inner_final:
        verts = hexagon_vertices((h[0], h[1]), 1.0, h[2])
        polygons.append(verts)
        
    max_proj = 0.0
    for i in range(6):
        theta = np.deg2rad(outer_angle_final + i * 60)
        nx = np.cos(theta)
        ny = np.sin(theta)
        for poly in polygons:
            projections = poly[:, 0] * nx + poly[:, 1] * ny
            max_p = np.max(projections)
            if max_p > max_proj:
                max_proj = max_p
                
    A_min = max_proj
    L = A_min * 2.0 / np.sqrt(3)
    
    # Add tiny margin for floating point safety
    L += 1e-6
    
    return inner_final, (0.0, 0.0), L, outer_angle_final

def construct():
    return construct_impl()
