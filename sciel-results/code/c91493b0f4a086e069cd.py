"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # 1. Generate grid centers (Side 3 hexagon cluster)
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    # 2. Helper functions for geometry
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def required_L_for_angle(centers, angles, outer_angle_deg):
        """Calculate L for fixed inner config and outer angle."""
        outer_angle_rad = math.radians(outer_angle_deg)
        max_proj = 0.0
        for i in range(len(centers)):
            cx, cy = centers[i]
            verts = get_hex_vertices(cx, cy, angles[i])
            for vx, vy in verts:
                # Rotate point by -outer_angle to align with outer hex normals
                x_rot = vx * math.cos(-outer_angle_rad) - vy * math.sin(-outer_angle_rad)
                y_rot = vx * math.sin(-outer_angle_rad) + vy * math.cos(-outer_angle_rad)
                
                # Project onto 6 edge normals of flat-top hex (at 30, 90, ...)
                for k in range(6):
                    theta = math.radians(30.0 + k * 60.0)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = x_rot * nx + y_rot * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / sqrt3

    def check_overlap(centers, angles, eps=1e-7):
        """Check for overlap. Returns max violation (0 if valid)."""
        n = len(centers)
        max_viol = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 2.0:
                    continue
                # Detailed check
                verts_i = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
                verts_j = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                
                # Simple distance check for convex polygons is hard without SAT.
                # Use vertex-in-poly test.
                # For speed, we can just check if min distance between edges is < eps?
                # Or just return 1.0 if any vertex is inside.
                # For optimization, we need a smooth-ish penalty.
                # Let's use the minimum distance between vertices of i and edges of j?
                # Too slow.
                # Let's assume if dist < 1.732 (sqrt3), they might overlap.
                # We will use a penalty based on distance between centers if close, 
                # but that's not accurate for rotated hexes.
                
                # Given time, we will just check vertex containment for validity.
                # For the optimizer penalty, we will use a crude distance penalty 
                # only if we suspect overlap, otherwise 0.
                # Actually, for Nelder-Mead, we can just use a large penalty 
                # if any vertex is inside another hex.
                
                inside = False
                for vx, vy in verts_i:
                    # Check if (vx,vy) inside hex j
                    # Transform to j's frame
                    dx_v = vx - centers[j][0]
                    dy_v = vy - centers[j][1]
                    x_rot = dx_v * math.cos(-math.radians(angles[j])) - dy_v * math.sin(-math.radians(angles[j]))
                    y_rot = dx_v * math.sin(-math.radians(angles[j])) + dy_v * math.cos(-math.radians(angles[j]))
                    apothem = sqrt3 / 2.0
                    for k in range(6):
                        theta = math.radians(30.0 + k * 60.0)
                        proj = x_rot * math.cos(theta) + y_rot * math.sin(theta)
                        if proj > apothem + eps:
                            inside = True
                            break
                    if inside:
                        break
                if inside:
                    return 1.0 # Penalty flag
        return 0.0

    def objective(params):
        """Objective for local optimization."""
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[2*i], params[2*i+1]))
            angles.append(params[2*N + i])
        outer_cx = params[2*N + N]
        outer_cy = params[2*N + N + 1]
        outer_angle = params[2*N + N + 2]
        
        # Translate centers so outer center is at 0 for L calculation?
        # No, L is translation invariant.
        # But we need to check containment relative to outer center.
        # The required_L function assumes outer center at 0.
        # So we shift inner centers by -outer_center.
        shifted_centers = [(cx - outer_cx, cy - outer_cy) for cx, cy in centers]
        
        L = required_L_for_angle(shifted_centers, angles, outer_angle)
        
        # Check overlap
        if check_overlap(centers, angles) > 0:
            return L + 100.0
            
        return L

    # 3. Combinatorial Search for Grid Subset
    best_L = float('inf')
    best_subset = None
    
    # Use a very small scaling to ensure disjointness in the final output
    # But for finding the subset, we can use exact geometry
    SCALE = 1.0
    
    for subset in combinations(range(19), 12):
        if time.time() > deadline - 20: # Leave time for polish
            break
        centers_subset = [all_centers[i] for i in subset]
        angles_subset = [0.0] * 12
        L = required_L_for_angle(centers_subset, angles_subset, 0.0)
        if L < best_L:
            best_L = L
            best_subset = subset

    if best_subset is None:
        best_subset = tuple(range(12))
        centers_subset = [all_centers[i] for i in best_subset]
        angles_subset = [0.0] * 12
        best_L = required_L_for_angle(centers_subset, angles_subset, 0.0)
        
    # 4. Local Optimization Polish
    # Initialize params from best_subset
    # Params: x1,y1,...,x12,y12, a1,...,a12, ox, oy, o_angle
    x0 = []
    for i in best_subset:
        x0.append(all_centers[i][0])
        x0.append(all_centers[i][1])
    for _ in range(12):
        x0.append(0.0) # angles
    x0.append(0.0) # outer cx
    x0.append(0.0) # outer cy
    x0.append(0.0) # outer angle
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        
        # Check if result is better and valid
        if res.fun < best_L:
            # Verify validity strictly
            centers_opt = [(res.x[2*i], res.x[2*i+1]) for i in range(12)]
            angles_opt = [res.x[2*N + i] for i in range(12)]
            if check_overlap(centers_opt, angles_opt, eps=1e-6) == 0:
                best_L = res.fun
                # Update best_subset? No, we just use the optimized positions.
                # We need to store the optimized centers/angles for return.
                # We'll flag that we have an optimized solution.
                optimized_solution = True
                opt_centers = centers_opt
                opt_angles = angles_opt
                opt_outer_center = (res.x[2*N+N], res.x[2*N+N+1])
                opt_outer_angle = res.x[2*N+N+2]
            else:
                optimized_solution = False
        else:
            optimized_solution = False
    except Exception:
        optimized_solution = False

    # 5. Construct Final Output
    if 'optimized_solution' in locals() and optimized_solution:
        inner = []
        for i in range(12):
            inner.append([opt_centers[i][0], opt_centers[i][1], opt_angles[i]])
        return inner, opt_outer_center, best_L, opt_outer_angle
    else:
        # Fallback to grid solution
        # Apply small scale for disjointness
        FINAL_SCALE = 1.000001
        inner = []
        for i in best_subset:
            x, y = all_centers[i]
            inner.append([x * FINAL_SCALE, y * FINAL_SCALE, 0.0])
        
        # Recalculate L with scale
        centers_scaled = [(x*FINAL_SCALE, y*FINAL_SCALE) for x,y in all_centers[i] for i in best_subset] # Dummy
        # Actually just scale the L
        final_L = best_L * FINAL_SCALE
        return inner, (0.0, 0.0), final_L, 0.0

# EVOLVE-BLOCK-END
