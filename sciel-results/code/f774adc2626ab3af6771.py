import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator limit 150s, keep 10s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (0,0) with rotation angle
    def get_hex_vertices_local(angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = math.cos(theta)
            y = math.sin(theta)
            verts.append((x, y))
        return verts

    # Precompute local vertices for all possible rotations? No, rotation is continuous.
    # But we can compute vertices on the fly.
    
    def get_hex_vertices(cx, cy, angle_deg):
        local_verts = get_hex_vertices_local(angle_deg)
        return [(cx + vx, cy + vy) for vx, vy in local_verts]

    def dist_point_segment(px, py, x1, y1, x2, y2):
        """Distance from point P to segment AB."""
        vx = x2 - x1
        vy = y2 - y1
        wx = px - x1
        wy = py - y1
        
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            return math.hypot(px - x1, py - y1)
            
        c2 = vx * vx + vy * vy
        if c2 <= c1:
            return math.hypot(px - x2, py - y2)
            
        t = c1 / c2
        proj_x = x1 + t * vx
        proj_y = y1 + t * vy
        return math.hypot(px - proj_x, py - proj_y)

    def min_dist_poly(verts1, verts2):
        """
        Calculate the minimum distance between two convex polygons.
        If they overlap, return 0.
        If they are disjoint, return the minimum distance between any vertex of one 
        and any edge of the other.
        """
        min_d = float('inf')
        
        # Check if any vertex of poly1 is inside poly2 or vice versa?
        # For convex polygons, if min distance is 0, they touch or overlap.
        # A simpler check for overlap in convex polygons is SAT, but we want a smooth distance.
        # The minimum distance between two convex polygons is the minimum of:
        # 1. Distances from vertices of poly1 to edges of poly2
        # 2. Distances from vertices of poly2 to edges of poly1
        # If the polygons overlap, this distance will be 0 (if we consider interior distance as 0).
        # However, `dist_point_segment` returns 0 if the point is on the segment.
        # If the point is inside the polygon, the distance to the boundary is > 0, but we want 0 for overlap.
        # So we need to check if a vertex is strictly inside the other polygon.
        
        # Efficient check: If the minimum distance calculated by vertex-edge is > 0, 
        # they are disjoint. If it is 0, they might touch or overlap.
        # To distinguish overlap from touch, we can check if any vertex is inside the other.
        
        # Let's compute min vertex-edge distance first.
        for i in range(6):
            x1, y1 = verts1[i]
            x2, y2 = verts1[(i + 1) % 6]
            for j in range(6):
                x3, y3 = verts2[j]
                d = dist_point_segment(x3, y3, x1, y1, x2, y2)
                if d < min_d:
                    min_d = d
                    if min_d == 0:
                        # Need to verify if it's an overlap or just touching.
                        # If touching, distance is 0. If overlapping, distance is 0 (in the sense of separation).
                        # For optimization, we want a penalty that grows as they get closer.
                        # If they overlap, we want a large penalty.
                        # Let's assume if min_d is 0, they are in contact. 
                        # To detect overlap, we can check if any vertex of poly1 is inside poly2.
                        pass
        
        # Check for overlap explicitly using point-in-polygon for a few vertices
        # If any vertex of poly1 is inside poly2, they overlap.
        # Point-in-convex-polygon test is fast.
        def point_in_poly(px, py, verts):
            # Cross product method for convex polygon
            sign = 0
            for i in range(6):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if cross != 0:
                    if sign == 0:
                        sign = 1 if cross > 0 else -1
                    elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                        return False
            return True

        # Check if any vertex of poly1 is inside poly2
        for i in range(6):
            if point_in_poly(verts1[i][0], verts1[i][1], verts2):
                return 0.0
        
        # Check if any vertex of poly2 is inside poly1
        for i in range(6):
            if point_in_poly(verts2[i][0], verts2[i][1], verts1):
                return 0.0

        # If no overlap, return the minimum distance
        return min_d

    def evaluate(p, outer_angle_deg=0.0):
        """
        p: array of length 3*N. [x1, y1, a1, x2, y2, a2, ...]
        Returns a smooth cost function.
        """
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        # Center the configuration
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        hexagons = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hexagons.append(get_hex_vertices(cx, cy, a))
        
        # Calculate collision penalty
        # We want a smooth function. If distance d < 0 (overlap), penalty is high.
        # Since min_dist_poly returns 0 for overlap, we need a different metric for overlap depth?
        # Or we can just use a large penalty if min_dist_poly == 0.
        # To make it smooth, we can define the cost as sum of max(0, overlap_depth)^2.
        # But calculating overlap depth is hard.
        # Alternative: Use a "soft" distance.
        # If they overlap, min_dist_poly is 0.
        # We can use a penalty based on the violation of the constraint.
        # Let's stick to: if min_dist_poly < epsilon, add penalty.
        # To make it differentiable-ish for NM, we can add a term that grows as distance approaches 0.
        # Actually, NM doesn't need gradients, but it benefits from a smooth landscape.
        # A common trick: penalty = sum( max(0, d_min - d)^2 ) where d_min is a small buffer?
        # No, we want them to be disjoint, so d >= 0.
        # If d == 0, they touch. We want d > 0.
        # Let's use penalty = sum( (1.0 / (d + 0.1))^2 ) for d < some threshold?
        # Or simply: if d == 0, penalty = 100. If d > 0, penalty = 0.
        # This is not smooth.
        
        # Better: Use the minimum distance itself.
        # If they overlap, min_dist_poly returns 0.
        # We can define the cost as:
        # For each pair, if min_dist_poly == 0, cost += 100.
        # This is a step function. NM might struggle.
        
        # Let's try a different approach for the objective:
        # Minimize the outer side length, subject to no overlap.
        # If we use a penalty method, we need to know the depth of overlap.
        # For convex polygons, the overlap depth can be approximated by the minimum translation distance (MTD).
        # Calculating MTD is more complex.
        
        # Let's go back to the hard constraint but use a smarter search.
        # The previous SA failed because the landscape was rugged due to the hard penalty.
        # If we use a "soft" penalty that is continuous, it might work better.
        
        # Let's define the cost as:
        # C = Outer_Side_Length + Lambda * Sum( max(0, Overlap_Depth)^2 )
        # We don't have Overlap_Depth.
        
        # Let's use a different metric: 
        # If the polygons overlap, the minimum distance is 0.
        # We can use the fact that if they are disjoint, the distance is > 0.
        # We can define a "virtual" distance that is negative when overlapping.
        # This requires solving for the MTD.
        
        # Given time constraints, let's stick to the hard penalty but use a more robust local search.
        # The key insight from the prompt is to use a "smooth, differentiable collision metric".
        # I will implement a simple approximation of the overlap penalty using the minimum distance.
        # If min_dist_poly returns 0, we assume they are in contact or overlapping.
        # To distinguish, we can check the area of intersection? Too slow.
        
        # Let's try this:
        # Cost = Outer_Side_Length
        # Constraint: min_dist_poly > 0
        # We will use a penalty function: P = Sum( max(0, 0.01 - min_dist_poly)^2 )
        # This penalizes being close to each other, effectively keeping them apart.
        # It is smooth and differentiable.
        
        collision_cost = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_poly(hexagons[i], hexagons[j])
                # If d is 0, they are touching/overlapping.
                # We want to push them apart.
                # If d < 0.01, add penalty.
                if d < 0.01:
                    collision_cost += (0.01 - d) ** 2
        
        # Calculate outer side length
        # For SA/NM, use fixed outer angle 0.0 for speed
        # Precompute axis directions for outer hexagon
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        side = 2.0 * max_proj / SQRT3
        
        # Weight the collision cost heavily
        # As side decreases, we want collision cost to be zero.
        # Let's use a fixed weight for now.
        return side + 10.0 * collision_cost

    # Axial to Cartesian for initial configurations
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Define initial configurations
    # Config 1: Flower + 4 extensions (known valid)
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    
    best_params = None
    best_score = float('inf')
    
    # Run Nelder-Mead from multiple starting points
    # We will run it for a limited time.
    
    for config_idx in range(3):
        if time.time() > deadline - 20.0:
            break
            
        # Use slightly perturbed initial configs to explore different basins
        params = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            # Add small noise to escape exact lattice
            x += random.gauss(0, 0.1)
            y += random.gauss(0, 0.1)
            a = random.gauss(0, 5.0)
            params.extend([x, y, a])
        params = np.array(params)
        
        current_score = evaluate(params)
        
        # Nelder-Mead
        try:
            res = minimize(evaluate, params, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass

    if best_params is None:
        # Fallback
        params = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)
        best_score = evaluate(best_params)

    # Final Refinement: Optimize Outer Angle
    final_params = best_params.copy()
    
    centers = []
    angles = []
    for i in range(N):
        centers.append((final_params[i*3], final_params[i*3+1]))
        angles.append(final_params[i*3+2])
        
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
    
    hexagons = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hexagons.append(get_hex_vertices(cx, cy, a))
        
    # Verify validity
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            d = min_dist_poly(hexagons[i], hexagons[j])
            if d == 0.0:
                # Check if they actually overlap or just touch
                # If they touch, it's valid (disjoint usually means no interior intersection)
                # The problem says "disjoint", which usually allows touching.
                # However, if they overlap, min_dist_poly is also 0.
                # We need to distinguish.
                # Let's assume if min_dist_poly is 0, we need to check overlap.
                # Re-use point_in_poly check
                v1 = hexagons[i]
                v2 = hexagons[j]
                def point_in_poly(px, py, verts):
                    sign = 0
                    for k in range(6):
                        x1, y1 = verts[k]
                        x2, y2 = verts[(k + 1) % 6]
                        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                        if cross != 0:
                            if sign == 0:
                                sign = 1 if cross > 0 else -1
                            elif (cross > 0 and sign < 0) or (cross < 0 and sign > 0):
                                return False
                    return True
                overlap = False
                for k in range(6):
                    if point_in_poly(v1[k][0], v1[k][1], v2):
                        overlap = True
                        break
                if not overlap:
                    for k in range(6):
                        if point_in_poly(v2[k][0], v2[k][1], v1):
                            overlap = True
                            break
                if overlap:
                    valid = False
                    break
        if not valid:
            break

    if not valid:
        # Fallback to initial valid config
        print("Warning: Final result invalid, falling back to initial config")
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        
        # Calculate side for fallback
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        max_proj = 0.0
        for hex_verts in hexagons_fb:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        best_side = 2.0 * max_proj / SQRT3
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Optimize outer angle
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + oa)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        side = 2.0 * max_proj / SQRT3
        if side < best_side:
            best_side = side
            best_oa = oa
            
    # Construct final inner list
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
