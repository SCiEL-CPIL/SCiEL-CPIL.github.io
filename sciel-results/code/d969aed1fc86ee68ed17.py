import math
import time
import numpy as np
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
    # Normals to the sides of the outer hexagon
    # A regular hexagon with side s has width 2*s*cos(30) = s*sqrt(3)
    # The distance from center to side is s * sqrt(3) / 2
    # So if max projection is P, then P = s * sqrt(3) / 2 => s = 2 * P / sqrt(3)
    
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate point into outer hexagon frame
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
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
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def generate_shape(axial_coords):
    """Convert axial coordinates to Cartesian centers."""
    centers = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Define candidate shapes (axial coordinates)
    # Shape 1: Symmetric "Flower" (Center + Ring + 4 Diagonals)
    shape1_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1)
    ]
    
    # Shape 2: Symmetric "Cross" (Center + Ring + 4 Cardinals at dist 2)
    shape2_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    # Shape 3: Parent's "3-4-3" (Center + Ring + 4 in a row)
    shape3_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    # Shape 4: Another compact variation
    shape4_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, -2), (0, -2)
    ]
    
    candidates = [shape1_axial, shape2_axial, shape3_axial, shape4_axial]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0, 0)
    best_outer_angle = 0.0
    
    # Phase 1: Evaluate each candidate shape with optimal outer hexagon alignment
    for idx, axial_coords in enumerate(candidates):
        if time.time() > deadline - 30:
            break
            
        centers = generate_shape(axial_coords)
        # Initialize inner hexagons with angle 0
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Optimize outer hexagon
        def objective_outer(params):
            cx, cy, angle_deg = params
            return compute_required_side(initial_inner, (cx, cy), angle_deg)
        
        best_val = float('inf')
        best_params = None
        
        # Try multiple starting angles
        for init_angle in [0, 15, 30, 45, 60]:
            if time.time() > deadline - 30:
                break
            init_params = [0, 0, float(init_angle)]
            try:
                res = minimize(objective_outer, init_params, method='Nelder-Mead',
                              options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_params is not None and best_val < best_side:
            best_side = best_val
            best_inner = initial_inner
            best_outer_center = (best_params[0], best_params[1])
            best_outer_angle = best_params[2]
            
    # Phase 2: Joint refinement of the best shape
    # Allow small perturbations of inner hexagons and outer hexagon
    # This helps if the lattice alignment is not perfectly optimal or if individual rotation helps
    
    if best_inner is not None:
        def objective_joint(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]
            
            side = compute_required_side(inner_list, (ocx, ocy), oangle)
            
            # Penalty for overlaps
            penalty = 0.0
            for i in range(N):
                vi = get_hex_vertices(inner_list[i][0], inner_list[i][1], 1.0, math.radians(inner_list[i][2]))
                for j in range(i + 1, N):
                    vj = get_hex_vertices(inner_list[j][0], inner_list[j][1], 1.0, math.radians(inner_list[j][2]))
                    if sat_overlap_check(vi, vj, 1e-7):
                        penalty += 100.0
            return side + penalty
        
        initial_params = []
        for x, y, a in best_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
        
        # Use Nelder-Mead for joint optimization
        # This is expensive, so we limit iterations
        try:
            res_joint = minimize(objective_joint, initial_params, method='Nelder-Mead',
                                options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 5000})
            
            if res_joint.fun < best_side:
                # Verify disjointness
                test_inner = []
                for i in range(N):
                    test_inner.append([res_joint.x[i*3], res_joint.x[i*3+1], res_joint.x[i*3+2]])
                
                if check_all_disjoint(test_inner, 1e-6):
                    best_side = res_joint.fun
                    best_inner = test_inner
                    best_outer_center = (res_joint.x[N*3], res_joint.x[N*3+1])
                    best_outer_angle = res_joint.x[N*3+2]
        except Exception:
            pass

    # Fallback if something went wrong
    if best_inner is None:
        centers = generate_shape(shape1_axial)
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_side = compute_required_side(best_inner, (0,0), 0.0)
        best_outer_center = (0,0)
        best_outer_angle = 0.0

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
