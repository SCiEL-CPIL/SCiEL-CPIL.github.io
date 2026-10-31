"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
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
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    This provides better gradient information for optimization.
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
    
    # Collect all projections
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
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
    
    # Log-sum-exp for smooth max
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    # Smooth max via log-sum-exp: (1/beta) * log(sum(exp(beta * p_i)))
    # This approximates max(p_i) as beta -> infinity
    scaled = [beta * (p - max_proj) for p in projections]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
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
        
        # Check for gap
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    # Outer hexagon vertices
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    # For each inner hexagon, check if all vertices are inside the outer hexagon
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            # Check if point is inside convex polygon using cross products
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                # Vector from p1 to p2
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                # Vector from p1 to point
                px = vx - p1[0]
                py = vy - p1[1]
                # Cross product
                cross = ex * py - ey * px
                # For a convex polygon with vertices in counter-clockwise order,
                # the point is inside if all cross products have the same sign
                # We assume the hexagon is oriented such that interior is to the left
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice - use tighter, more symmetric arrangement
    # Center hexagon + 6 around it + 4 more in a compact cluster
    # Axial coordinates for hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian with tight packing
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon using scipy with multiple starts
    # Use smooth surrogate for better optimization landscape
    
    def objective_outer_smooth(params):
        cx, cy, angle_deg = params
        side = compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)
        return side
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        side = compute_required_side(initial_inner, (cx, cy), angle_deg)
        return side
    
    # Initial guess: centroid and 0
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')
    
    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    
    for init_angle in init_angles:
        if time.time() > deadline - 20:
            break
            
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            # First pass: smooth optimization for basin finding
            res_smooth = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'maxfev': 1000})
            
            # Second pass: exact optimization
            res_exact = minimize(objective_outer_exact, res_smooth.x, method='Nelder-Mead', 
                                 options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
            
            if res_exact.fun < best_val:
                best_val = res_exact.fun
                best_result = res_exact.x
        except Exception:
            pass
    
    if best_result is not None:
        ox, oy, oa = best_result
        best_outer_center = (ox, oy)
        best_outer_angle = oa % 360
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)
        
        if check_validity(initial_inner, best_outer_center, best_side, best_outer_angle):
            best_inner = initial_inner
        else:
            # If invalid, try to fix by adjusting scale
            # This is a fallback, ideally we should optimize the inner positions too
            best_inner = initial_inner
            best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback to initial configuration
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)
        best_inner = initial_inner

    # 4. Optimize inner positions to reduce overlaps and improve packing
    # This is a more complex optimization with 3*N variables
    
    def objective_inner(params):
        if time.time() > deadline:
            return float('inf')
        
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2] * 180.0 / math.pi
            inner.append([x, y, ang])
        
        # Compute required side length
        side = compute_required_side(inner, best_outer_center, best_outer_angle)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(N):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            for j in range(i + 1, N):
                vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
                if sat_overlap_check(vi, vj, 1e-6):
                    # Calculate distance between centers for penalty
                    dx = inner[i][0] - inner[j][0]
                    dy = inner[i][1] - inner[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    penalty += max(0, 2.0 - dist) ** 2 * 100.0
        
        return side + penalty
    
    # Initialize inner parameters
    inner_params = np.zeros(3 * N)
    for i, (x, y, ang) in enumerate(best_inner):
        inner_params[3*i] = x
        inner_params[3*i+1] = y
        inner_params[3*i+2] = math.radians(ang)
    
    try:
        res_inner = minimize(objective_inner, inner_params, method='L-BFGS-B', 
                            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        
        if res_inner.fun < best_side + 0.1:  # Allow some tolerance
            inner_new = []
            for i in range(N):
                x = res_inner.x[3*i]
                y = res_inner.x[3*i+1]
                ang = res_inner.x[3*i+2] * 180.0 / math.pi
                inner_new.append([x, y, ang])
            
            side_new = compute_required_side(inner_new, best_outer_center, best_outer_angle)
            
            if check_validity(inner_new, best_outer_center, side_new, best_outer_angle):
                if side_new < best_side:
                    best_inner = inner_new
                    best_side = side_new
    except Exception:
        pass

    # 5. Final refinement of outer hexagon with optimized inner
    if time.time() < deadline - 10:
        def objective_final_outer(params):
            cx, cy, angle_deg = params
            side = compute_required_side(best_inner, (cx, cy), angle_deg)
            return side
        
        try:
            res_final = minimize(objective_final_outer, np.array([best_outer_center[0], best_outer_center[1], best_outer_angle]), 
                                method='Nelder-Mead', 
                                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
            
            if res_final.fun < best_side:
                ox, oy, oa = res_final.x
                side_final = compute_required_side(best_inner, (ox, oy), oa)
                
                if check_validity(best_inner, (ox, oy), side_final, oa % 360):
                    best_outer_center = (ox, oy)
                    best_outer_angle = oa % 360
                    best_side = side_final
        except Exception:
            pass

    # Ensure the result is valid
    if not check_validity(best_inner, best_outer_center, best_side, best_outer_angle):
        # Fallback to a simple valid configuration
        # Use the initial lattice with a larger outer hexagon
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle) + 0.1
        best_inner = initial_inner

    return (best_inner, best_outer_center, best_side, best_outer_angle)

# EVOLVE-BLOCK-END
