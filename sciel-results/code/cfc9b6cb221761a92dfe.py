import numpy as np
from scipy.optimize import minimize
import time
import math

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # Safety margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def hexagons_overlap(v1, v2, eps=1e-6):
    """
    Check if two hexagons (defined by their 6 vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap.
    """
    # Pre-compute axes from edges of v1
    axes = []
    
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Pre-compute axes from edges of v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Project all vertices once per polygon for each axis
    # To optimize, we can project v1 and v2 for all axes first
    # But simpler: just loop through axes
    
    for axis in axes:
        ax, ay = axis
        
        # Project v1
        min1 = float('inf')
        max1 = float('-inf')
        for p in v1:
            proj = p[0] * ax + p[1] * ay
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        # Project v2
        min2 = float('inf')
        max2 = float('-inf')
        for p in v2:
            proj = p[0] * ax + p[1] * ay
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        # Check for gap. If there is a gap on any axis, they don't overlap.
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
    return True

def is_layout_valid(inner, eps=1e-5):
    """Check if all inner hexagons are disjoint, allowing for a small epsilon gap."""
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        verts_list.append(verts)
    
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if hexagons_overlap(verts_list[i], verts_list[j], eps):
                return False
    return True

def compute_overlap_penalty(inner, eps=1e-5):
    """
    Compute a smooth penalty for overlaps.
    Returns 0 if no overlap, positive value otherwise.
    Uses a simple sum of penetration depths (approximated).
    """
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        verts_list.append(verts)
    
    n = len(verts_list)
    total_penalty = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            v1 = verts_list[i]
            v2 = verts_list[j]
            
            # Quick distance check using centers
            cx1 = sum(p[0] for p in v1) / 6
            cy1 = sum(p[1] for p in v1) / 6
            cx2 = sum(p[0] for p in v2) / 6
            cy2 = sum(p[1] for p in v2) / 6
            
            dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
            # Minimum distance between centers for non-overlap is roughly 2 * apothem = 2 * sqrt(3)/2 = sqrt(3)
            # But with rotation, it varies. A safe bound is distance between centers > 2 * (max radius) = 2.
            # Actually, for unit hexagons, the minimum center distance for non-overlap is sqrt(3) if aligned.
            # If dist_sq is large, they definitely don't overlap.
            if dist_sq > 4.0: # 2^2, safe upper bound for non-overlap
                continue
            
            # Check overlap
            # We need a measure of overlap. 
            # Simple approach: if they overlap, add a penalty based on how much they overlap.
            # For simplicity in this penalty function, we just check if they overlap and add a fixed penalty.
            # A better way would be to compute the minimum distance between edges, but that's expensive.
            # Let's stick to a binary penalty for now, or use the SAT gap.
            
            # Re-implement SAT to get the maximum gap (negative if overlap)
            axes = []
            for i_idx in range(6):
                p1 = v1[i_idx]
                p2 = v1[(i_idx + 1) % 6]
                edge = (p2[0] - p1[0], p2[1] - p1[1])
                normal = (-edge[1], edge[0])
                norm = math.sqrt(normal[0]**2 + normal[1]**2)
                if norm > 1e-9:
                    axes.append((normal[0] / norm, normal[1] / norm))
            
            for i_idx in range(6):
                p1 = v2[i_idx]
                p2 = v2[(i_idx + 1) % 6]
                edge = (p2[0] - p1[0], p2[1] - p1[1])
                normal = (-edge[1], edge[0])
                norm = math.sqrt(normal[0]**2 + normal[1]**2)
                if norm > 1e-9:
                    axes.append((normal[0] / norm, normal[1] / norm))
            
            min_gap = float('inf')
            
            for axis in axes:
                ax, ay = axis
                min1 = float('inf')
                max1 = float('-inf')
                min2 = float('inf')
                max2 = float('-inf')
                
                for p in v1:
                    proj = p[0] * ax + p[1] * ay
                    if proj < min1: min1 = proj
                    if proj > max1: max1 = proj
                    
                for p in v2:
                    proj = p[0] * ax + p[1] * ay
                    if proj < min2: min2 = proj
                    if proj > max2: max2 = proj
                
                gap = min(max1 - min2, max2 - min1)
                if gap < min_gap:
                    min_gap = gap
            
            if min_gap < 0:
                # Overlap. Penalty is proportional to depth of overlap
                total_penalty += (-min_gap) ** 2
    
    return total_penalty

def compute_required_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    
    # Normals for a hexagon with angle 0
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle))
    sin_a = math.sin(math.radians(outer_angle))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate to align with outer hexagon axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle, beta=10.0):
    """
    Compute a smooth approximation of the minimum outer side length.
    Uses log-sum-exp to approximate the max projection.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    
    # Normals for a hexagon with angle 0
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Collect all projections
    projections = []
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle))
    sin_a = math.sin(math.radians(outer_angle))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate to align with outer hexagon axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    # Log-sum-exp for max
    max_proj = max(projections)
    if max_proj > 1000: # Avoid overflow
        return 2.0 * max_proj / SQRT3
        
    sum_exp = 0.0
    for p in projections:
        sum_exp += math.exp(beta * (p - max_proj))
    
    max_approx = max_proj + (1.0 / beta) * math.log(sum_exp)
    
    return 2.0 * max_approx / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # 1. Define 11 hexagons on a hexagonal lattice
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    # Use a scale factor > 1 to ensure initial non-overlap with a safe margin.
    # Standard lattice spacing for side 1 is sqrt(3) for y and 1.5 for x.
    # Scaling by 1.01 adds a small gap.
    scale = 1.01
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    # Verify validity. If invalid (should not happen with scale 1.01), fall back to a safe linear arrangement.
    if not is_layout_valid(initial_inner):
        initial_inner = [[i * 3.0, 0.0, 0.0] for i in range(N)]
        centers = [(i * 3.0, 0.0) for i in range(N)]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try a range of angles and optimize center for each
    for outer_angle_deg in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
        if time.time() > deadline - 30:
            break
            
        # Optimize center for this angle
        def obj_center(params):
            return compute_required_side(best_inner, (params[0], params[1]), outer_angle_deg)
            
        try:
            res = minimize(
                obj_center,
                [cx_avg, cy_avg],
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_outer_center = (res.x[0], res.x[1])
                best_outer_angle = outer_angle_deg
        except Exception:
            pass

    # 4. Joint Optimization with Smooth Surrogate
    # We will optimize the positions of all inner hexagons and the outer hexagon parameters simultaneously.
    # Parameters: [x1, y1, a1, ..., x11, y11, a11, cx, cy, outer_angle]
    # Total 11*3 + 3 = 36 parameters.
    
    if time.time() < deadline - 15:
        current_inner = [list(x) for x in best_inner]
        current_cx, current_cy = best_outer_center
        current_angle = best_outer_angle
        
        # Flatten to vector
        def flatten(inner, cx, cy, angle):
            vec = []
            for h in inner:
                vec.extend(h)
            vec.extend([cx, cy, angle])
            return vec
            
        def unflatten(vec):
            inner = []
            for i in range(N):
                inner.append(list(vec[i*3:i*3+3]))
            cx = vec[N*3]
            cy = vec[N*3+1]
            angle = vec[N*3+2]
            return inner, cx, cy, angle
            
        x0 = flatten(current_inner, current_cx, current_cy, current_angle)
        
        def objective(vec):
            inner, cx, cy, angle = unflatten(vec)
            
            # Check validity (hard constraint)
            if not is_layout_valid(inner):
                # Return a large penalty if invalid
                # Also add overlap penalty to guide back to feasibility
                penalty = compute_overlap_penalty(inner)
                return 100.0 + penalty
            
            # Compute smooth side length
            side = compute_required_side_smooth(inner, (cx, cy), angle, beta=10.0)
            return side
        
        try:
            # Use Nelder-Mead for the joint optimization
            # It doesn't require gradients, which is good for the non-differentiable validity constraint.
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            
            if res.fun < best_side:
                # Check if the result is valid
                res_inner, res_cx, res_cy, res_angle = unflatten(res.x)
                if is_layout_valid(res_inner):
                    # Verify with exact side calculation
                    exact_side = compute_required_side(res_inner, (res_cx, res_cy), res_angle)
                    if exact_side < best_side:
                        best_side = exact_side
                        best_inner = [list(h) for h in res_inner]
                        best_outer_center = (res_cx, res_cy)
                        best_outer_angle = res_angle
        except Exception:
            pass

    # 5. Local Search with Strict Validity Check
    # We will perform a simple local search. At each step, we try to move one hexagon slightly.
    # If the move results in an overlap, it is rejected. If it reduces the outer side length, it is accepted.
    # This is slower than gradient-based methods but much more robust for hard constraints.
    
    if time.time() < deadline - 5:
        current_inner = [list(x) for x in best_inner]
        current_cx, current_cy = best_outer_center
        current_angle = best_outer_angle
        current_side = best_side
        
        step_size = 0.1
        iterations = 0
        max_iterations = 5000
        
        while time.time() < deadline - 2 and iterations < max_iterations:
            iterations += 1
            improved = False
            
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                    
                # Try moving hexagon i in random directions
                for _ in range(5):
                    dx = (np.random.rand() - 0.5) * step_size
                    dy = (np.random.rand() - 0.5) * step_size
                    da = (np.random.rand() - 0.5) * 0.1 # Small rotation in radians, convert to deg later if needed, but inner uses deg
                    
                    new_inner = [list(x) for x in current_inner]
                    new_inner[i][0] += dx
                    new_inner[i][1] += dy
                    new_inner[i][2] += da * 180.0 / math.pi
                    
                    if is_layout_valid(new_inner):
                        # Compute new side length
                        # Optimize center quickly for this configuration? 
                        # For speed, we can just re-center on the centroid or use the current center.
                        # Re-centering on centroid is a good heuristic.
                        cx_new = sum(h[0] for h in new_inner) / N
                        cy_new = sum(h[1] for h in new_inner) / N
                        
                        side_new = compute_required_side(new_inner, (cx_new, cy_new), current_angle)
                        
                        if side_new < current_side - 1e-7:
                            current_inner = new_inner
                            current_cx, current_cy = cx_new, cy_new
                            current_side = side_new
                            best_inner = [list(x) for x in current_inner]
                            best_outer_center = (current_cx, current_cy)
                            best_side = current_side
                            improved = True
                            break # Restart loop for this hexagon if improved
                            
            if not improved and iterations % 100 == 0:
                step_size *= 0.95 # Decrease step size if no improvement

    # Final validation
    if not is_layout_valid(best_inner):
        # Fallback to initial lattice
        best_inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
        best_side = compute_required_side(best_inner, (0.0, 0.0), 0.0)
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0

    final_side = best_side + 1e-9
    return best_inner, best_outer_center, final_side, best_outer_angle
