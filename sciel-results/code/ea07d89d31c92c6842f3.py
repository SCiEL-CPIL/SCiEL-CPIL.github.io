import numpy as np
from scipy.optimize import minimize, differential_evolution
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

def hexagons_overlap(v1, v2, eps=1e-7):
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
        if max1 < min2 + eps or max2 < min1 + eps:
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

    # 4. Joint Optimization: Optimize inner positions/angles AND outer parameters
    # Strategy: Treat the problem as minimizing the outer side length.
    # We will use Nelder-Mead on a reduced set of variables or a multi-stage approach.
    
    # To handle the non-convexity and constraints (disjointness), we use a penalty method.
    # However, for speed, we will focus on fine-tuning the initial lattice packing.
    # The lattice packing is locally optimal for aligned hexagons.
    # Rotating individual hexagons can sometimes improve packing if the outer boundary is tight,
    # but for 11 hexagons in a 19-hexagon hole (or similar), the lattice is usually very good.
    
    # Let's try a local search that allows small rotations and translations.
    # We optimize: [x1, y1, a1, ..., x11, y11, a11, cx, cy, outer_angle]
    
    def objective_joint(params):
        if time.time() > deadline - 5:
            return 1e10
        
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner.append([x, y, a])
        
        cx = params[3*N]
        cy = params[3*N+1]
        outer_angle = params[3*N+2]
        
        # Check validity
        if not is_layout_valid(inner):
            # Compute penalty
            penalty = 0.0
            verts_list = []
            for (ix, iy, iangle) in inner:
                verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
                verts_list.append(verts)
            
            for i in range(N):
                for j in range(i + 1, N):
                    if hexagons_overlap(verts_list[i], verts_list[j]):
                        # Approximate penalty
                        penalty += 100.0
                        break
                if penalty > 100:
                    break
            
            return compute_required_side(inner, (cx, cy), outer_angle) + penalty
        
        return compute_required_side(inner, (cx, cy), outer_angle)

    # Initial parameters
    init_params = []
    for x, y, a in best_inner:
        init_params.append(x)
        init_params.append(y)
        init_params.append(a)
    init_params.append(best_outer_center[0])
    init_params.append(best_outer_center[1])
    init_params.append(best_outer_angle)

    # Run Nelder-Mead on the full joint space
    # This is a 35D problem (11*3 + 3). Nelder-Mead can struggle in high dimensions,
    # but the landscape is relatively smooth near the lattice solution.
    
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective_joint,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7, 'maxfev': 1000}
            )
            
            if result.fun < best_side + 1e-4:
                new_inner = []
                for i in range(N):
                    new_inner.append([result.x[3*i], result.x[3*i+1], result.x[3*i+2]])
                
                if is_layout_valid(new_inner):
                    new_cx = result.x[3*N]
                    new_cy = result.x[3*N+1]
                    new_angle = result.x[3*N+2]
                    
                    side = compute_required_side(new_inner, (new_cx, new_cy), new_angle)
                    if side < best_side:
                        best_side = side
                        best_inner = new_inner
                        best_outer_center = (new_cx, new_cy)
                        best_outer_angle = new_angle
        except Exception:
            pass

    # 5. Try Differential Evolution for a global search if time permits
    # DE is robust but slow. We'll use a small population and few iterations.
    if time.time() < deadline - 40:
        try:
            bounds = []
            for i in range(N):
                bounds.append((-5, 5))
                bounds.append((-5, 5))
                bounds.append((-180, 180))
            bounds.append((-5, 5)) # cx
            bounds.append((-5, 5)) # cy
            bounds.append((-180, 180)) # outer angle
            
            de_result = differential_evolution(
                objective_joint,
                bounds,
                maxiter=20,
                popsize=10,
                tol=1e-4,
                seed=42,
                polish=False
            )
            
            if de_result.fun < best_side + 1e-4:
                new_inner = []
                for i in range(N):
                    new_inner.append([de_result.x[3*i], de_result.x[3*i+1], de_result.x[3*i+2]])
                
                if is_layout_valid(new_inner):
                    new_cx = de_result.x[3*N]
                    new_cy = de_result.x[3*N+1]
                    new_angle = de_result.x[3*N+2]
                    
                    side = compute_required_side(new_inner, (new_cx, new_cy), new_angle)
                    if side < best_side:
                        best_side = side
                        best_inner = new_inner
                        best_outer_center = (new_cx, new_cy)
                        best_outer_angle = new_angle
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
