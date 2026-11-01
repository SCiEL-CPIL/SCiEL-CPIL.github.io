"""Hexagon packing: 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Outward normals for standard hexagon (vertices at 0, 60, ...):
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def build_compact_layout():
    """
    Build a compact arrangement of 11 hexagons.
    Uses a hexagonal lattice with slight compression to allow optimization to find
    a tighter packing.
    """
    # Axial coordinates for a compact cluster of 11 hexagons
    # Center + ring of 6 + 4 more to fill gaps
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # Convert axial to Cartesian with tight packing
    # For hexagonal lattice: x = 1.5*q, y = sqrt(3)*(r + q/2)
    # The minimum center-to-center distance for non-overlapping unit hexagons
    # with aligned faces is sqrt(3) ≈ 1.732
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
    
    # All hexagons start aligned at 0 degrees
    layout = [[x, y, 0.0] for x, y in centers]
    
    return layout

def optimize_composite(inner_layout, deadline):
    """
    Optimize the composite of inner hexagons and outer container.
    Treats all 11 hexagons as a composite object, optimizing:
    1. Individual positions and angles of inner hexagons
    2. Outer container center and angle
    
    Uses a smooth surrogate for the containment constraint and
    a penalty for overlaps.
    """
    n = len(inner_layout)
    
    # Flatten parameters: 3 per inner hexagon (x, y, angle) + 3 for outer (cx, cy, angle)
    def flatten(inner_list, outer_center, outer_angle):
        params = []
        for ix, iy, ia in inner_list:
            params.extend([ix, iy, ia])
        params.extend([outer_center[0], outer_center[1], outer_angle])
        return params
    
    def unflatten(params):
        inner_list = []
        for i in range(n):
            inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
        outer_center = (params[n*3], params[n*3+1])
        outer_angle = params[n*3+2]
        return inner_list, outer_center, outer_angle
    
    def composite_objective(params, use_smooth=True, beta=60.0):
        inner_list, outer_center, outer_angle = unflatten(params)
        
        if use_smooth:
            side = compute_required_side_smooth(inner_list, outer_center, outer_angle, beta)
        else:
            side = compute_required_side(inner_list, outer_center, outer_angle)
        
        # Overlap penalty using center distance heuristic
        # Minimum center distance for non-overlapping aligned hexagons: sqrt(3)
        # For rotated hexagons, the minimum distance can be smaller, but we use
        # a conservative bound
        penalty = 0.0
        min_dist = SQRT3 - 0.02  # Slight margin to allow optimization to work
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + 500.0 * penalty
    
    # Initial parameters
    cx_avg = sum(c[0] for c in inner_layout) / n
    cy_avg = sum(c[1] for c in inner_layout) / n
    init_params = flatten(inner_layout, (cx_avg, cy_avg), 0.0)
    
    best_params = init_params.copy()
    best_val = composite_objective(best_params, use_smooth=False)
    
    # Multi-start optimization with different outer angles
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    for init_angle in init_angles:
        if time.time() > deadline - 30:
            break
        
        trial_params = init_params.copy()
        trial_params[n*3+2] = float(init_angle)
        
        try:
            # Phase 1: Smooth optimization for basin finding
            res_smooth = minimize(
                lambda p: composite_objective(p, use_smooth=True, beta=60.0),
                trial_params,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1500}
            )
            
            # Phase 2: Exact optimization from smooth result
            if res_smooth.fun < best_val or best_params is None:
                res_exact = minimize(
                    lambda p: composite_objective(p, use_smooth=False),
                    res_smooth.x,
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 3000}
                )
                
                if res_exact.fun < best_val:
                    best_val = res_exact.fun
                    best_params = res_exact.x.copy()
                    
        except Exception:
            pass
    
    # Unflatten best result
    best_inner, best_outer_center, best_outer_angle = unflatten(best_params)
    best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

def polish_solution(inner, outer_center, outer_angle, deadline):
    """
    High-precision polish pass to refine the solution.
    """
    n = len(inner)
    
    def flatten(inner_list, outer_center, outer_angle):
        params = []
        for ix, iy, ia in inner_list:
            params.extend([ix, iy, ia])
        params.extend([outer_center[0], outer_center[1], outer_angle])
        return params
    
    def unflatten(params):
        inner_list = []
        for i in range(n):
            inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
        outer_center = (params[n*3], params[n*3+1])
        outer_angle = params[n*3+2]
        return inner_list, outer_center, outer_angle
    
    def polish_objective(params):
        inner_list, outer_center, outer_angle = unflatten(params)
        side = compute_required_side(inner_list, outer_center, outer_angle)
        
        # Tighter overlap penalty
        penalty = 0.0
        min_dist = SQRT3 - 0.01
        for i in range(n):
            for j in range(i + 1, n):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + 2000.0 * penalty
    
    init_params = flatten(inner, outer_center, outer_angle)
    
    try:
        result = minimize(
            polish_objective,
            init_params,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 10000}
        )
        
        if result.fun < compute_required_side(inner, outer_center, outer_angle):
            polished_inner, polished_center, polished_angle = unflatten(result.x)
            if check_all_disjoint(polished_inner, eps=1e-6):
                polished_side = compute_required_side(polished_inner, polished_center, polished_angle)
                return polished_inner, polished_center, polished_side, polished_angle
    except Exception:
        pass
    
    return inner, outer_center, compute_required_side(inner, outer_center, outer_angle), outer_angle

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Build initial compact layout
    initial_layout = build_compact_layout()
    
    # 2. Optimize composite (inner + outer)
    best_inner, best_outer_center, best_side, best_outer_angle = optimize_composite(
        initial_layout, deadline
    )
    
    # 3. Verify with SAT
    if not check_all_disjoint(best_inner, eps=1e-5):
        # Fallback to initial layout if SAT fails
        best_inner = initial_layout
        cx_avg = sum(c[0] for c in best_inner) / len(best_inner)
        cy_avg = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # 4. Polish if time permits
    if time.time() < deadline - 10:
        best_inner, best_outer_center, best_side, best_outer_angle = polish_solution(
            best_inner, best_outer_center, best_outer_angle, deadline
        )
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
