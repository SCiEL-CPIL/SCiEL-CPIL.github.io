import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Precompute unit hexagon vertices (flat-top, side=1, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    # Vertices: (1, 0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1, 0), (-0.5, -sqrt(3)/2), (0.5, -sqrt(3)/2)
    angles = np.deg2rad(np.arange(6) * 60)
    unit_verts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normals for the 6 faces of the outer hexagon (flat-top)
    # Outer hexagon with side S has "inradius" (distance to face) R = S * sqrt(3) / 2
    # The support function for a point p in direction n is dot(p, n)
    # For a flat-top hexagon centered at origin, the 6 outward normals are at angles 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(30 + np.arange(6) * 60)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Min distance between centers for non-overlapping unit hexagons
    # For two unit hexagons, if they are not rotated relative to each other, min center dist is sqrt(3).
    # With rotation, the minimum distance can be smaller (if they interlock) or larger.
    # Actually, for regular hexagons, the minimum distance between centers for disjointness depends on rotation.
    # The "worst case" (smallest distance allowing overlap) is when a vertex of one hits a face of the other?
    # No, we want to ensure they are disjoint.
    # The condition for two convex polygons to be disjoint is that there exists a separating axis.
    # For our objective, we need to check if they overlap.
    # A simpler constraint: min distance between any vertex of A and any edge of B > 0 and vice versa?
    # This is expensive.
    # Approximation: Use the minimum distance between centers as a proxy?
    # If center distance < sqrt(3), they might overlap. If > 2, they definitely don't.
    # Actually, the minimum distance between centers for two non-overlapping unit hexagons is sqrt(3) (when aligned).
    # If rotated, they can be closer? No, the "width" is sqrt(3).
    # Let's use a robust check: compute the minimum distance between the two polygons.
    
    def min_dist_polygons(v1, v2):
        """
        Compute minimum distance between two convex polygons (hexagons).
        Returns 0 if they intersect.
        v1, v2: (6, 2) arrays of vertices.
        """
        min_d = float('inf')
        # Check vertex of v1 to edges of v2
        for i in range(6):
            p1 = v1[i]
            for j in range(6):
                p2 = v2[j]
                p3 = v2[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        # Check vertex of v2 to edges of v1
        for i in range(6):
            p1 = v2[i]
            for j in range(6):
                p2 = v1[j]
                p3 = v1[(j+1)%6]
                v = p3 - p2
                w = p1 - p2
                c1 = np.dot(w, v)
                c2 = np.dot(v, v)
                if c1 <= 0:
                    d = np.linalg.norm(p1 - p2)
                elif c1 >= c2:
                    d = np.linalg.norm(p1 - p3)
                else:
                    d = np.linalg.norm(w - (c1/c2)*v)
                if d < min_d:
                    min_d = d
        return min_d

    def objective(params):
        """
        params: 36 values.
        [0:12]   x positions
        [12:24]  y positions
        [24:36]  rotation angles (radians)
        
        Returns: (outer_side_length, constraint_violation)
        We minimize outer_side_length + penalty * constraint_violation
        """
        xs = params[0:12]
        ys = params[12:24]
        rots = params[24:36]
        
        # Calculate vertices for all 12 hexagons
        verts_list = []
        for i in range(12):
            c = np.array([xs[i], ys[i]])
            r = rots[i]
            # Rotate unit vertices
            cos_r = np.cos(r)
            sin_r = np.sin(r)
            rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            v = unit_verts @ rot_mat.T + c
            verts_list.append(v)
            
        # 1. Calculate required outer side length
        # The outer hexagon is centered at (0,0) with side S.
        # Inradius R = S * sqrt(3) / 2 => S = 2 * R / sqrt(3)
        # The constraint is that for every vertex p of every inner hex,
        # max(dot(p, n_k)) <= R for all 6 normals n_k.
        # So R_min = max_over_all_vertices( max_k(dot(p, n_k)) )
        
        max_support = 0.0
        for v in verts_list:
            # v is (6,2). normals is (6,2).
            # We need max of dot(v[i], normals[j]) for all i,j
            # This is equivalent to max of dot(v[i], n) for n in normals
            # Since normals are fixed, we can compute dot products for all vertices and all normals
            # (12*6 vertices) x (6 normals)
            # Let's do it in a vectorized way for speed if possible, but loop is fine for 72 points
            for p in v:
                dots = np.dot(normals, p)
                m = np.max(dots)
                if m > max_support:
                    max_support = m
        
        # Ensure max_support is positive (it should be, as hexagons have size)
        if max_support < 1e-6:
             max_support = 1e-6
             
        outer_S = 2.0 * max_support / np.sqrt(3)
        
        # 2. Check overlap constraints
        # We want min_dist > 0.
        # Penalty for overlap: sum of (epsilon - min_dist)^2 for pairs where min_dist < epsilon
        # Or simply: if min_dist < 0, it's an intersection.
        # Since min_dist_polygons returns 0 if intersecting, we can't distinguish "touching" from "intersecting".
        # We need a margin. Let's say we want min_dist >= 0.
        # If min_dist is 0, they are touching or intersecting.
        # To ensure disjointness, we need min_dist > 0.
        # However, numerical precision makes > 0 hard.
        # Let's use a small epsilon, e.g., 1e-4.
        # Penalty: sum of max(0, eps - min_dist)^2
        
        eps = 1e-3
        overlap_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                # Quick check: center distance
                cx1, cy1 = xs[i], ys[i]
                cx2, cy2 = xs[j], ys[j]
                c_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                # Minimum possible distance between centers for unit hexagons is sqrt(3) ~ 1.732
                # If c_dist < sqrt(3) - eps, they MUST overlap?
                # Actually, if rotated, they can be closer?
                # The "width" of a unit hexagon is sqrt(3).
                # So if center distance < sqrt(3), they might overlap.
                # If center distance > 2, they definitely don't.
                # Let's just compute min_dist if c_dist < 2.0
                if c_dist < 2.0:
                    d = min_dist_polygons(verts_list[i], verts_list[j])
                    if d < eps:
                        # Penalty
                        overlap_penalty += (eps - d) ** 2
        
        # Total objective
        # We want to minimize outer_S.
        # If there is overlap, we add a large penalty.
        # To make DE work well, we can return outer_S + 100 * overlap_penalty
        # But if overlap is large, outer_S might be small, but penalty huge.
        # This is a standard penalty method.
        
        return outer_S + 100.0 * overlap_penalty

    # Define bounds for DE
    # x, y: [-5, 5] (should be enough for S ~ 4-5)
    # rot: [0, 2*pi]
    bounds = [(-5, 5)] * 12 + [(-5, 5)] * 12 + [(0, 2*np.pi)] * 12
    
    # Initial population
    # DE will handle this.
    
    # Run Differential Evolution
    # maxiter: limit to avoid timeout
    # popsize: 15 (default is 15*len(bounds) which is huge, 540. Let's use 20)
    # tol: 1e-4
    # seed: for reproducibility? No, let it be random.
    
    try:
        # Use a wrapper to capture the best solution
        # DE returns the best x and fun
        
        # To save time, we can limit the number of function evaluations.
        # 120s limit.
        # Each eval involves 66 min_dist checks (worst case) + vertex calcs.
        # min_dist is ~ 72 ops.
        # Should be fast.
        
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=20, 
            tol=1e-5, 
            seed=42, # Fixed seed for stability in this context? 
            # Actually, random seed might explore better. But fixed is safer for debugging.
            # Let's use a random seed to allow different runs to potentially find different minima.
            # But for a single construct() call, fixed is fine.
            # Wait, the task says "Each evaluation must finish within 120s".
            # DE with popsize=20, 36 dims, 100 iters = 2000 evals.
            # If each eval is ~1ms, that's 2s. Very fast.
            # We can increase maxiter.
            # Let's try maxiter=300.
        )
        
        if res.success or res.fun < 10.0: # If we found something reasonable
            best_params = res.x
            best_score = res.fun
        else:
            # Fallback to a known good configuration?
            # Or just return the best found even if not "success"
            best_params = res.x
            best_score = res.fun
            
    except Exception as e:
        # Fallback
        # Simple grid
        xs = np.zeros(12)
        ys = np.zeros(12)
        rots = np.zeros(12)
        # Arrange in 2 rows of 6?
        # Side length of hex is 1.
        # Horizontal spacing: 1.5? No, flat-top hex width is 2, height is sqrt(3).
        # If aligned, horizontal step is 1.5? No.
        # Center to center for adjacent flat-top hexes (sharing vertical edge? No, flat-top shares horizontal edges? No.)
        # Flat-top hex: Vertices at (1,0), (0.5, h), (-0.5, h), (-1,0), (-0.5,-h), (0.5,-h)
        # h = sqrt(3)/2.
        # Two such hexes side-by-side (sharing vertical edge? No, they share an edge if offset by 1.5 in x and sqrt(3)/2 in y? No.)
        # Let's just use a loose grid.
        for i in range(12):
            row = i // 4
            col = i % 4
            xs[i] = col * 2.1 - 3.0
            ys[i] = row * 1.8 - 1.5
            rots[i] = 0.0
        best_params = np.concatenate([xs, ys, rots])
        best_score = objective(best_params)

    # Extract best solution
    xs = best_params[0:12]
    ys = best_params[12:24]
    rots = best_params[24:36]
    
    # Convert rotations to degrees
    rots_deg = np.rad2deg(rots)
    
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], rots_deg[i]])
        
    # Calculate the actual outer side length for the returned solution
    # We need to be careful: the objective included a penalty.
    # We should calculate the true outer side length based on the geometry, ignoring overlap penalty
    # BUT the solution must be valid (no overlaps).
    # If there are overlaps, the score is invalid?
    # "score is the outer side length"
    # "an invalid result ... scores 0"
    # So if overlaps exist, we should try to fix them or return a valid one.
    # DE minimizes the penalized objective. If penalty is 0, it's valid.
    # If penalty > 0, it's invalid.
    
    # Let's check validity of the best_params
    # Recalculate penalty
    penalty_part = 0.0
    # We need to re-evaluate to get the pure outer_S and penalty
    # Let's write a helper to get (S, penalty)
    
    def get_metrics(params):
        xs = params[0:12]
        ys = params[12:24]
        rots = params[24:36]
        
        verts_list = []
        for i in range(12):
            c = np.array([xs[i], ys[i]])
            r = rots[i]
            cos_r = np.cos(r)
            sin_r = np.sin(r)
            rot_mat = np.array([[cos_r, -sin_r], [sin_r, cos_r]])
            v = unit_verts @ rot_mat.T + c
            verts_list.append(v)
            
        max_support = 0.0
        for v in verts_list:
            for p in v:
                dots = np.dot(normals, p)
                m = np.max(dots)
                if m > max_support:
                    max_support = m
        
        outer_S = 2.0 * max_support / np.sqrt(3)
        
        eps = 1e-3
        overlap_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                cx1, cy1 = xs[i], ys[i]
                cx2, cy2 = xs[j], ys[j]
                c_dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if c_dist < 2.0:
                    d = min_dist_polygons(verts_list[i], verts_list[j])
                    if d < eps:
                        overlap_penalty += (eps - d) ** 2
        
        return outer_S, overlap_penalty

    true_S, true_pen = get_metrics(best_params)
    
    if true_pen > 1e-4:
        # The solution has overlaps. 
        # This is bad. We should try to find a valid one.
        # Since DE minimizes S + 100*Pen, if Pen is non-zero, S might be small but invalid.
        # We should ideally run DE with a constraint or a higher penalty weight.
        # Or, we can use the result from DE, but if it's invalid, we can try to "repair" it?
        # Or just return a known valid configuration with a larger S?
        # A valid configuration: 12 hexagons in a 4x3 grid?
        # S ~ 4 or 5.
        # Let's check if there's a valid solution in the DE history? 
        # DE doesn't store history easily.
        
        # Let's try to increase the penalty weight and re-run for a short time?
        # Or just fallback to a safe, valid layout.
        
        # Safe layout: 
        # 3 rows of 4?
        # Or 2 rows of 6?
        # Let's use a grid that is definitely valid.
        # Spacing: x_step = 2.0, y_step = 1.75 (sqrt(3) ~ 1.732)
        # If we use x_step=2.1, y_step=1.8, they are disjoint.
        
        xs_safe = np.zeros(12)
        ys_safe = np.zeros(12)
        rots_safe = np.zeros(12)
        for i in range(12):
            row = i // 4
            col = i % 4
            xs_safe[i] = col * 2.1 - 3.0
            ys_safe[i] = row * 1.8 - 1.5
            rots_safe[i] = 0.0
        safe_params = np.concatenate([xs_safe, ys_safe, rots_safe])
        safe_S, safe_pen = get_metrics(safe_params)
        
        if safe_pen < 1e-4:
            # Use safe solution
            best_params = safe_params
            true_S = safe_S
        else:
            # Even safe solution has overlap? Unlikely with 2.1/1.8 spacing.
            # If so, we have a bug in min_dist or geometry.
            # But 2.1 > 2 (max width), so they are definitely disjoint.
            # So safe_pen should be 0.
            pass

    # Final extraction
    xs = best_params[0:12]
    ys = best_params[12:24]
    rots = best_params[24:36]
    rots_deg = np.rad2deg(rots)
    
    inner = []
    for i in range(12):
        inner.append([xs[i], ys[i], rots_deg[i]])
        
    return inner, (0.0, 0.0), true_S, 0.0
