import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

N = 11
SQRT3 = math.sqrt(3)

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def hexagons_overlap(v1, v2):
    """
    Check if two hexagons (defined by their 6 vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap.
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal to edge
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = float('inf')
        max1 = float('-inf')
        min2 = float('inf')
        max2 = float('-inf')
        
        for p in v1:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        for p in v2:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        # Check for gap
        if max1 < min2 or max2 < min1:
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
            if hexagons_overlap(verts_list[i], verts_list[j]):
                return False
    return True

def compute_required_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    
    # Normals for a hexagon with angle 0 (flat top/bottom? No, vertex at 0 is pointy right).
    # If angle=0, vertices are at 0, 60, 120...
    # The sides are perpendicular to these.
    # The outward normals for the sides of a hexagon centered at origin with side S and rotation 0:
    # The vertices are at angles 0, 60, ...
    # The edges connect v0-v1, v1-v2, etc.
    # The normal to the edge between v0(1,0) and v1(0.5, sqrt3/2) is perpendicular to (v1-v0).
    # v1-v0 = (-0.5, sqrt3/2). Normal = (sqrt3/2, 0.5) which is angle 30.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
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
            # If outer hex is rotated by alpha, we rotate points by -alpha to check against standard hex
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

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
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    if not is_layout_valid(initial_inner):
        initial_inner = [[i * 2.5, 0.0, 0.0] for i in range(N)]
        centers = [(i * 2.5, 0.0) for i in range(N)]

    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon
    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
        side = compute_required_side(initial_inner, (0.0, 0.0), outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg

    # Optimize outer center
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    current_cx = cx_avg
    current_cy = cy_avg
    
    step = 0.5
    for _ in range(2):
        improved = False
        for dx in np.linspace(-step, step, 5):
            for dy in np.linspace(-step, step, 5):
                oc = (current_cx + dx, current_cy + dy)
                side = compute_required_side(best_inner, oc, best_outer_angle)
                if side < best_side - 1e-5:
                    best_side = side
                    best_outer_center = oc
                    current_cx = oc[0]
                    current_cy = oc[1]
                    improved = True
        if not improved:
            break
        step *= 0.5

    # 4. Continuous optimization of inner hexagon positions
    # We'll use Nelder-Mead to optimize positions while maintaining non-overlap
    # by using a penalty function for overlaps
    
    def compute_overlap_penalty(inner, penalty_weight=1000.0):
        """Compute penalty for overlapping hexagons."""
        verts_list = []
        for (ix, iy, iangle) in inner:
            verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
            verts_list.append(verts)
        
        n = len(verts_list)
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if hexagons_overlap(verts_list[i], verts_list[j]):
                    # Compute minimum distance between the two hexagons
                    # Approximate by checking vertex-to-vertex distances
                    min_dist = float('inf')
                    for v1 in verts_list[i]:
                        for v2 in verts_list[j]:
                            d = math.sqrt((v1[0]-v2[0])**2 + (v1[1]-v2[1])**2)
                            if d < min_dist:
                                min_dist = d
                    # If they overlap, min_dist should be 0 or very small
                    # We use a quadratic penalty based on how much they overlap
                    if min_dist < 0.1:
                        penalty += penalty_weight * (0.1 - min_dist) ** 2
                    else:
                        penalty += penalty_weight * 0.01
        return penalty

    def objective(params):
        """Objective function for optimization: minimize outer side + overlap penalty."""
        if time.time() > deadline - 10:
            return 1e10
        
        inner = []
        for i in range(N):
            x = params[2*i]
            y = params[2*i+1]
            inner.append([x, y, 0.0])
        
        # Check for overlaps
        overlap_pen = compute_overlap_penalty(inner)
        if overlap_pen > 1e-3:
            return 1e6 + overlap_pen
        
        # Compute required side length
        side = compute_required_side(inner, best_outer_center, best_outer_angle)
        return side + overlap_pen

    # Initial parameters from current best
    init_params = []
    for x, y, _ in best_inner:
        init_params.append(x)
        init_params.append(y)
    
    # Try Nelder-Mead optimization
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            
            if result.fun < best_side + 1e-3:
                new_inner = []
                for i in range(N):
                    new_inner.append([result.x[2*i], result.x[2*i+1], 0.0])
                
                if is_layout_valid(new_inner):
                    # Recompute best outer center and angle
                    temp_best_side = float('inf')
                    temp_best_angle = 0.0
                    temp_best_center = best_outer_center
                    
                    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
                        side = compute_required_side(new_inner, best_outer_center, outer_angle_deg)
                        if side < temp_best_side:
                            temp_best_side = side
                            temp_best_angle = outer_angle_deg
                    
                    # Optimize outer center for new inner layout
                    def obj_oc(params):
                        oc = (params[0], params[1])
                        return compute_required_side(new_inner, oc, temp_best_angle)
                    
                    res_oc = minimize(
                        obj_oc,
                        [best_outer_center[0], best_outer_center[1]],
                        method='Nelder-Mead',
                        options={'maxiter': 100}
                    )
                    if res_oc.fun < temp_best_side:
                        temp_best_side = res_oc.fun
                        temp_best_center = (res_oc.x[0], res_oc.x[1])
                    
                    if temp_best_side < best_side:
                        best_side = temp_best_side
                        best_inner = new_inner
                        best_outer_center = temp_best_center
                        best_outer_angle = temp_best_angle
        except Exception:
            pass

    # 5. Try differential evolution for global search if time permits
    if time.time() < deadline - 30:
        try:
            # Define bounds for DE
            bounds = []
            for i in range(N):
                bounds.append((-5, 5))
                bounds.append((-5, 5))
            
            de_result = differential_evolution(
                objective,
                bounds,
                maxiter=50,
                popsize=15,
                tol=1e-4,
                seed=42,
                polish=False
            )
            
            if de_result.fun < best_side + 1e-3:
                new_inner = []
                for i in range(N):
                    new_inner.append([de_result.x[2*i], de_result.x[2*i+1], 0.0])
                
                if is_layout_valid(new_inner):
                    temp_best_side = float('inf')
                    temp_best_angle = 0.0
                    temp_best_center = best_outer_center
                    
                    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
                        side = compute_required_side(new_inner, best_outer_center, outer_angle_deg)
                        if side < temp_best_side:
                            temp_best_side = side
                            temp_best_angle = outer_angle_deg
                    
                    def obj_oc2(params):
                        oc = (params[0], params[1])
                        return compute_required_side(new_inner, oc, temp_best_angle)
                    
                    res_oc = minimize(
                        obj_oc2,
                        [best_outer_center[0], best_outer_center[1]],
                        method='Nelder-Mead',
                        options={'maxiter': 100}
                    )
                    if res_oc.fun < temp_best_side:
                        temp_best_side = res_oc.fun
                        temp_best_center = (res_oc.x[0], res_oc.x[1])
                    
                    if temp_best_side < best_side:
                        best_side = temp_best_side
                        best_inner = new_inner
                        best_outer_center = temp_best_center
                        best_outer_angle = temp_best_angle
        except Exception:
            pass

    # Final validation
    if not is_layout_valid(best_inner):
        # Fallback to initial lattice
        best_inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
        best_side = compute_required_side(best_inner, (0.0, 0.0), 0.0)
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0

    final_side = best_side + 1e-9
    return best_inner, best_outer_center, final_side, best_outer_angle
