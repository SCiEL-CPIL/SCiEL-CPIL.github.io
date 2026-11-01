"""Packing 11 unit hexagons in a regular hexagon - discrete topology search."""
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

def axial_to_cartesian(q, r):
    """Convert axial hex coordinates to Cartesian (for unit hexagons)."""
    x = 1.5 * q
    y = SQRT3 * (r + q / 2.0)
    return (x, y)

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
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner)
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts_list.append(hex_vertices(ix, iy, 1.0, math.radians(iangle)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def generate_polyhex_topologies(n_cells=11, max_radius=3):
    """
    Generate all connected polyhex topologies with n_cells cells using BFS.
    Returns list of sets of axial coordinates (normalized so min q,r is at origin).
    """
    if n_cells == 1:
        return [{(0, 0)}]
    
    # Generate all polyhexes up to n_cells using the "free polyhex" generation
    # We use a canonical form to avoid duplicates
    
    def canonicalize(cells):
        """Return canonical form of a set of axial coordinates."""
        cells = set(cells)
        # Normalize: shift so that min q is 0, then min r is 0
        min_q = min(q for q, r in cells)
        min_r = min(r for q, r in cells if q == min_q)
        normalized = set()
        for q, r in cells:
            nq = q - min_q
            nr = r - min_r
            normalized.add((nq, nr))
        
        # Try all 6 rotations + 6 reflections, pick lexicographically smallest
        def rotate60(cells_set):
            # Rotation by 60 degrees in axial coords
            # (q, r) -> (-r, q+r)
            return set((-r, q + r) for q, r in cells_set)
        
        def reflect(cells_set):
            # Reflection: (q, r) -> (-q, -r)
            return set((-q, -r) for q, r in cells_set)
        
        def normalize_set(cells_set):
            min_q = min(q for q, r in cells_set)
            min_r = min(r for q, r in cells_set if q == min_q)
            return tuple(sorted((q - min_q, r - min_r) for q, r in cells_set))
        
        best = None
        current = set(normalized)
        for _ in range(6):
            for variant in [current, reflect(current)]:
                norm = normalize_set(variant)
                if best is None or norm < best:
                    best = norm
            current = rotate60(current)
        
        return best
    
    # Generate all polyhexes using the standard algorithm
    # Start with single cell, grow by adding adjacent cells
    # Use canonical form to avoid duplicates
    
    all_polyhexes = set()
    # Start with single cell
    all_polyhexes.add(canonicalize({(0, 0)}))
    
    current_size = 1
    while current_size < n_cells:
        next_polyhexes = set()
        for poly in all_polyhexes:
            if len(poly) != current_size:
                continue
            # Convert tuple back to set
            cells = set(poly)
            # Find all boundary cells that can be added
            candidates = set()
            for q, r in cells:
                # 6 neighbors
                for dq, dr in [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) not in cells:
                        candidates.add((nq, nr))
            
            for cand in candidates:
                new_cells = cells | {cand}
                canon = canonicalize(new_cells)
                next_polyhexes.add(canon)
        
        all_polyhexes.update(next_polyhexes)
        current_size += 1
    
    # Filter to exactly n_cells
    result = [tuple(sorted(p)) for p in all_polyhexes if len(p) == n_cells]
    return result

def get_neighbor_offsets():
    """Return the 6 neighbor offsets in axial coordinates."""
    return [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]

def generate_lattice_configs():
    """Generate multiple compact hex lattice configurations for 11 hexagons."""
    configs = []
    
    # Config 1: Flower + 4 extensions
    flower_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    configs.append(flower_coords)
    
    # Config 2: Row-based 4-3-4
    row_434 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    configs.append(row_434)
    
    # Config 3: 3-5-3 arrangement
    row_353 = [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    configs.append(row_353)
    
    # Config 4: 5-3-3 arrangement
    row_533 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    configs.append(row_533)
    
    # Config 5: Variant flower
    flower_v2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (2, -1), (0, 2)
    ]
    configs.append(flower_v2)
    
    # Config 6: Compact diamond-like
    diamond = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-1, 2), (1, 1), (0, 2)
    ]
    configs.append(diamond)
    
    # Config 7: 2-3-4-2 arrangement
    row_2342 = [
        (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3)
    ]
    configs.append(row_2342)
    
    # Config 8: 3-4-4 arrangement
    row_344 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    configs.append(row_344)
    
    # Config 9: 4-4-3 arrangement
    row_443 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    configs.append(row_443)
    
    # Config 10: 2-4-5 arrangement
    row_245 = [
        (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    configs.append(row_245)
    
    # Config 11: 5-4-2 arrangement
    row_542 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (2, 2), (3, 2)
    ]
    configs.append(row_542)
    
    # Config 12: 3-3-5 arrangement
    row_335 = [
        (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)
    ]
    configs.append(row_335)
    
    # Config 13: 5-3-3 variant
    row_533_v2 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2)
    ]
    configs.append(row_533_v2)
    
    # Config 14: 4-3-4 variant
    row_434_v2 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2), (3, 2)
    ]
    configs.append(row_434_v2)
    
    return configs

def optimize_outer_for_inner(inner, time_limit=5.0):
    """
    Optimize the outer hexagon (center and angle) for a fixed inner configuration.
    Returns (best_side, best_center, best_angle).
    """
    start_time = time.time()
    
    def objective(params):
        cx, cy, angle_deg = params
        return compute_required_side(inner, (cx, cy), angle_deg)
    
    best_val = float('inf')
    best_params = None
    
    # Try multiple initial angles
    init_angles = [0, 30, 60, 90, 120, 150, 15, 45, 75, 105, 135, 165]
    
    for init_angle in init_angles:
        if time.time() - start_time > time_limit:
            break
        
        init_params = [0.0, 0.0, float(init_angle)]
        
        try:
            res = minimize(objective, init_params, method='Nelder-Mead',
                          options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 10000})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if best_params is None:
        # Fallback
        best_val = compute_required_side(inner, (0.0, 0.0), 0.0)
        best_params = [0.0, 0.0, 0.0]
    
    return best_val, (best_params[0], best_params[1]), best_params[2]

def joint_optimize(inner_init, outer_init, time_limit=10.0):
    """
    Joint optimization of inner positions/rotations and outer hexagon.
    inner_init: list of [x, y, angle_deg]
    outer_init: (cx, cy, angle_deg)
    Returns (inner, outer_center, outer_angle, side).
    """
    start_time = time.time()
    n_inner = len(inner_init)
    
    def joint_objective(params):
        inner = []
        for i in range(n_inner):
            ix = params[i*3]
            iy = params[i*3 + 1]
            iangle = params[i*3 + 2]
            inner.append((ix, iy, iangle))
        
        ocx = params[n_inner*3]
        ocy = params[n_inner*3 + 1]
        oangle = params[n_inner*3 + 2]
        
        side = compute_required_side(inner, (ocx, ocy), oangle)
        
        # Penalty for overlapping inner hexagons
        penalty = 0.0
        for i in range(n_inner):
            for j in range(i+1, n_inner):
                dx = inner[i][0] - inner[j][0]
                dy = inner[i][1] - inner[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = 2.0
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2 * 50.0
        
        return side + penalty
    
    init_params = []
    for (ix, iy, iangle) in inner_init:
        init_params.extend([ix, iy, iangle])
    init_params.extend([outer_init[0], outer_init[1], outer_init[2]])
    
    best_val = joint_objective(init_params)
    best_params = list(init_params)
    
    try:
        res = minimize(joint_objective, init_params, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 15000})
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # Extract results
    new_inner = []
    for i in range(n_inner):
        new_inner.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
    
    new_ocx = best_params[n_inner*3]
    new_ocy = best_params[n_inner*3 + 1]
    new_oangle = best_params[n_inner*3 + 2]
    
    # Verify no overlaps
    if not check_all_disjoint(new_inner, eps=1e-5):
        # Revert to initial if overlaps
        return inner_init, outer_init, compute_required_side(inner_init, outer_init, outer_init[2])
    
    side = compute_required_side(new_inner, (new_ocx, new_ocy), new_oangle)
    return new_inner, (new_ocx, new_ocy), new_oangle, side

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Generate lattice configurations
    configs = generate_lattice_configs()
    
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Process each configuration
    for coords in configs:
        if time.time() > deadline - 30:
            break
        
        # Create Cartesian positions
        points = []
        for q, r in coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            points.append((x, y))
        
        # Center the configuration
        cx_avg = sum(p[0] for p in points) / len(points)
        cy_avg = sum(p[1] for p in points) / len(points)
        centered_points = [(p[0] - cx_avg, p[1] - cy_avg) for p in points]
        
        # Initial layout: All hexagons aligned (angle 0)
        initial_inner = [[x, y, 0.0] for x, y in centered_points]
        
        # Optimize outer hexagon
        side, center, angle = optimize_outer_for_inner(initial_inner, time_limit=3.0)
        
        if side < best_side:
            best_side = side
            best_inner = initial_inner
            best_outer_center = center
            best_outer_angle = angle
    
    # If no valid arrangement was found, use the first config as fallback
    if best_inner is None:
        points = []
        for q, r in configs[0]:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            points.append((x, y))
        cx_avg = sum(p[0] for p in points) / len(points)
        cy_avg = sum(p[1] for p in points) / len(points)
        centered_points = [(p[0] - cx_avg, p[1] - cy_avg) for p in points]
        best_inner = [[x, y, 0.0] for x, y in centered_points]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    # Joint optimization to improve the best configuration
    if time.time() < deadline - 25:
        new_inner, new_center, new_angle, new_side = joint_optimize(
            best_inner, (best_outer_center[0], best_outer_center[1], best_outer_angle),
            time_limit=15.0
        )
        
        if new_side < best_side:
            best_inner = new_inner
            best_outer_center = new_center
            best_outer_angle = new_angle
            best_side = new_side
    
    # Final high-precision polish on outer parameters only
    if time.time() < deadline - 5:
        def objective_outer_polish(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner, (cx, cy), angle_deg)
        
        best_polish_val = best_side
        best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
        
        # Try a few starting points for polish
        polish_starts = [
            [best_outer_center[0], best_outer_center[1], best_outer_angle],
            [best_outer_center[0] + 0.01, best_outer_center[1], best_outer_angle + 1.0],
            [best_outer_center[0], best_outer_center[1] + 0.01, best_outer_angle - 1.0],
            [best_outer_center[0] - 0.01, best_outer_center[1], best_outer_angle + 2.0],
            [best_outer_center[0], best_outer_center[1] - 0.01, best_outer_angle - 2.0],
        ]
        
        for ps in polish_starts:
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 25000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass
        
        if best_polish_params is not None:
            best_outer_center = (best_polish_params[0], best_polish_params[1])
            best_outer_angle = best_polish_params[2]
            best_side = best_polish_val
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
