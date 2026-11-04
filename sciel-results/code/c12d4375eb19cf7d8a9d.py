"""Optimized hexagon packing: flower pattern with rotation/translation optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 15.0
    
    # Unit hexagon vertices (pointy-top, vertices at 0°, 60°, 120°, 180°, 240°, 300°)
    def hex_vertices(cx, cy, angle=0.0):
        """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle."""
        angles = np.array([0, 60, 120, 180, 240, 300]) * math.pi / 180.0 + angle
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    # Standard flower pattern centers (in hex lattice coordinates)
    # Center at origin, 6 neighbors at distance sqrt(3), 4 specific ring-2 positions
    d = math.sqrt(3)  # distance between adjacent hex centers in honeycomb
    
    # Ring 1: 6 neighbors
    ring1 = []
    for i in range(6):
        angle = i * math.pi / 3
        ring1.append((d * math.cos(angle), d * math.sin(angle)))
    
    # Ring 2: 12 possible positions, we pick 4 that form the optimal pattern
    # Using axial coordinates to generate ring 2
    ring2_all = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_all.append((x, y))
    
    # The optimal pattern uses 4 specific ring-2 positions
    # These are at angles approximately 30°, 90°, 150°, 210° from center
    # Let's identify them: they should be the ones that create a symmetric pattern
    # For a 11-hexagon cluster, we want 4 ring-2 hexes that maintain symmetry
    # Try the configuration where ring-2 hexes are at (2,0), (0,2), (-2,0), (0,-2) in axial
    # which corresponds to specific Cartesian positions
    
    # Actually, let me just use all 12 ring-2 and pick the best 4 via optimization
    # But for speed, let's use a known good pattern: the "flower" with 4 ring-2 at 90° intervals
    
    # Generate all 11 centers for the standard flower
    # Center + 6 ring1 + 4 ring2
    # The 4 ring2 positions that work well: indices that give good symmetry
    # Let's try: (2,0), (1,1), (-1,1), (-2,0) in axial -> but need to verify
    
    # Simpler: just use the first 4 ring2 positions and optimize
    # Or better: try multiple patterns and pick best
    
    base_centers = [(0.0, 0.0)] + ring1
    
    # Try several ring-2 selections
    candidate_ring2_sets = []
    
    # Pattern 1: every other ring-2 position (4 out of 12, symmetric)
    # Ring 2 positions in order around the circle
    # Sort ring2_all by angle
    ring2_sorted = sorted(ring2_all, key=lambda p: math.atan2(p[1], p[0]))
    
    # Pick 4 evenly spaced: indices 0, 3, 6, 9
    pattern1 = [ring2_sorted[i] for i in [0, 3, 6, 9]]
    candidate_ring2_sets.append(pattern1)
    
    # Pattern 2: indices 1, 4, 7, 10
    pattern2 = [ring2_sorted[i] for i in [1, 4, 7, 10]]
    candidate_ring2_sets.append(pattern2)
    
    # Pattern 3: indices 2, 5, 8, 11
    pattern3 = [ring2_sorted[i] for i in [2, 5, 8, 11]]
    candidate_ring2_sets.append(pattern3)
    
    # For each pattern, optimize rotation and translation
    best_S = float('inf')
    best_result = None
    
    for pattern_idx, ring2_set in enumerate(candidate_ring2_sets):
        if time.time() > deadline - safety_margin:
            break
        
        all_centers = base_centers + ring2_set
        
        # Precompute all vertices for this pattern (66 points)
        # We'll rotate and translate the whole cluster
        
        # Function to compute the enclosing hexagon side length
        def compute_side_length(params):
            """params = [rotation_rad, tx, ty]"""
            rot = params[0]
            tx = params[1]
            ty = params[2]
            
            # Rotate all centers around origin, then translate
            cos_r, sin_r = math.cos(rot), math.sin(rot)
            
            # All vertices: for each center, 6 vertices
            # Total 66 points
            points = []
            for cx, cy in all_centers:
                # Rotate center
                rcx = cx * cos_r - cy * sin_r
                rcy = cx * sin_r + cy * cos_r
                # Translate
                rcx += tx
                rcy += ty
                # Add 6 vertices (also rotated)
                for i in range(6):
                    va = i * math.pi / 3 + rot
                    vx = math.cos(va)
                    vy = math.sin(va)
                    points.append((rcx + vx, rcy + vy))
            
            points_arr = np.array(points)
            
            # Find minimum enclosing regular hexagon
            # A regular hexagon with center (cx, cy), side S, rotation theta
            # has 6 supporting lines. The distance from center to each facet is S*sqrt(3)/2
            # For the hexagon to contain all points:
            # For each of 6 directions (theta + k*pi/3), max |P . n| <= S * sqrt(3)/2
            # where n is the unit normal
            
            # We need to find theta (container rotation) that minimizes S
            # S = (2/sqrt(3)) * max_k max_i |P_i . n_k|
            
            # Sweep container rotation theta
            # The function is periodic with 60 degrees
            # Sample theta from 0 to pi/3
            
            # For efficiency, sample many theta values
            theta_vals = np.linspace(0, math.pi / 3, 200, endpoint=False)
            
            min_S = float('inf')
            for theta in theta_vals:
                # 6 normal directions
                normals = []
                for k in range(6):
                    angle = theta + k * math.pi / 3
                    normals.append((math.cos(angle), math.sin(angle)))
                normals_arr = np.array(normals)  # (6, 2)
                
                # Project all points onto each normal
                # points_arr: (66, 2), normals_arr: (6, 2)
                # dots: (66, 6)
                dots = points_arr @ normals_arr.T  # (66, 6)
                abs_dots = np.abs(dots)
                
                # For each normal direction, max over points
                max_per_normal = np.max(abs_dots, axis=0)  # (6,)
                
                # Max over all normals
                max_proj = np.max(max_per_normal)
                
                S = (2.0 / math.sqrt(3)) * max_proj
                if S < min_S:
                    min_S = S
            
            return min_S
        
        # Optimize rotation, tx, ty
        # Initial guess: rotation=0, tx=0, ty=0
        x0 = [0.0, 0.0, 0.0]
        
        # Try multiple initial rotations
        best_local_S = float('inf')
        best_local_params = None
        
        for init_rot in [0, math.pi/6, math.pi/3, math.pi/4, math.pi/2]:
            if time.time() > deadline - safety_margin:
                break
            
            x0 = [init_rot, 0.0, 0.0]
            
            try:
                result = minimize(compute_side_length, x0, method='Nelder-Mead',
                                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                                         'adaptive': True})
                if result.fun < best_local_S:
                    best_local_S = result.fun
                    best_local_params = result.x
            except Exception:
                pass
        
        if best_local_params is not None and best_local_S < best_S:
            best_S = best_local_S
            rot, tx, ty = best_local_params
            best_result = (rot, tx, ty)
    
    # If optimization didn't find anything, use a fallback
    if best_result is None:
        # Use simple pattern with no rotation
        rot, tx, ty = 0.0, 0.0, 0.0
        # Compute S for this
        def compute_side_simple():
            points = []
            for cx, cy in base_centers + candidate_ring2_sets[0]:
                for i in range(6):
                    va = i * math.pi / 3
                    points.append((cx + math.cos(va), cy + math.sin(va)))
            points_arr = np.array(points)
            theta_vals = np.linspace(0, math.pi / 3, 100, endpoint=False)
            min_S = float('inf')
            for theta in theta_vals:
                normals = []
                for k in range(6):
                    angle = theta + k * math.pi / 3
                    normals.append((math.cos(angle), math.sin(angle)))
                normals_arr = np.array(normals)
                dots = points_arr @ normals_arr.T
                abs_dots = np.abs(dots)
                max_per_normal = np.max(abs_dots, axis=0)
                max_proj = np.max(max_per_normal)
                S = (2.0 / math.sqrt(3)) * max_proj
                if S < min_S:
                    min_S = S
            return min_S
        best_S = compute_side_simple()
    
    # Format output
    rot, tx, ty = best_result
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    
    inner = []
    for cx, cy in base_centers + candidate_ring2_sets[0]:
        # Rotate and translate
        rcx = cx * cos_r - cy * sin_r + tx
        rcy = cx * sin_r + cy * cos_r + ty
        inner.append([rcx, rcy, math.degrees(rot)])
    
    # Container center and side
    # The container should be centered at (tx, ty) in the rotated frame
    # Actually, we need to find the optimal container center
    # For simplicity, use (0,0) as container center, but this may not be optimal
    # Better: compute the centroid of all vertices and use that
    
    # Compute all vertices in final position
    all_verts = []
    for cx, cy in base_centers + candidate_ring2_sets[0]:
        rcx = cx * cos_r - cy * sin_r + tx
        rcy = cx * sin_r + cy * cos_r + ty
        for i in range(6):
            va = i * math.pi / 3 + rot
            all_verts.append((rcx + math.cos(va), rcy + math.sin(va)))
    
    # Container center: use (0,0) for now, or compute bounding box center
    # For a regular hexagon container, the optimal center is where the max projection is minimized
    # This is complex; use (0,0) as approximation
    container_center = (0.0, 0.0)
    
    # Recompute S with container at (0,0)
    points_arr = np.array(all_verts)
    # Find best container rotation
    theta_vals = np.linspace(0, math.pi / 3, 500, endpoint=False)
    min_S_final = float('inf')
    best_theta_final = 0.0
    for theta in theta_vals:
        normals = []
        for k in range(6):
            angle = theta + k * math.pi / 3
            normals.append((math.cos(angle), math.sin(angle)))
        normals_arr = np.array(normals)
        dots = points_arr @ normals_arr.T
        abs_dots = np.abs(dots)
        max_per_normal = np.max(abs_dots, axis=0)
        max_proj = np.max(max_per_normal)
        S = (2.0 / math.sqrt(3)) * max_proj
        if S < min_S_final:
            min_S_final = S
            best_theta_final = theta
    
    return inner, container_center, min_S_final, math.degrees(best_theta_final)
# EVOLVE-BLOCK-END
