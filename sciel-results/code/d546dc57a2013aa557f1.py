import math
import time
import random
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

def check_all_fit(inner, outer_center, S, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons fit inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, S, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            if not point_in_polygon(v, outer_verts):
                return False
    return True

def point_in_polygon(p, polygon):
    """Ray casting algorithm for point in polygon."""
    x, y = p
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """
    Compute a smooth penalty for overlaps based on center distances.
    This is a proxy for actual overlap, valid for aligned hexagons.
    Vectorized for speed.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    # Extract positions
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    # Compute all pairwise squared distances
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    # Only consider i < j pairs (upper triangle)
    iu = np.triu_indices(n, k=1)
    dists_sq_upper = dists_sq[iu]
    
    # Penalty: sum of (min_dist - dist)^2 where dist < min_dist
    dists = np.sqrt(dists_sq_upper)
    diff = np.maximum(0.0, min_dist - dists)
    penalty = np.sum(diff * diff)
    
    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_valid = False

    # Generate multiple initial configurations on hexagonal lattices
    def generate_lattice_config(spacing, angle_deg):
        """Generate N points on a triangular lattice, centered at origin."""
        # Triangular lattice basis
        a1 = (spacing, 0)
        a2 = (spacing / 2, spacing * SQRT3 / 2)
        
        max_coord = 8
        points = []
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                x = i * a1[0] + j * a2[0]
                y = i * a1[1] + j * a2[1]
                points.append((x, y))
        
        # Sort by distance from origin
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Take first N points
        selected = points[:N]
        
        # Rotate the entire configuration
        rad = math.radians(angle_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        
        # Center at centroid
        cx = sum(p[0] for p in selected) / N
        cy = sum(p[1] for p in selected) / N
        
        inner = []
        for x, y in selected:
            rx = (x - cx) * ca - (y - cy) * sa
            ry = (x - cx) * sa + (y - cy) * ca
            inner.append([rx, ry, 0.0])
        
        return inner

    def optimize_outer(inner, cx_init, cy_init, angles_to_try):
        """Optimize outer hexagon parameters for given inner configuration."""
        def objective_outer(params):
            cx, cy, angle_deg = params
            side = compute_required_side(inner, (cx, cy), angle_deg)
            # Add penalty if hexagons don't fit
            if not check_all_fit(inner, (cx, cy), side, angle_deg):
                return side + 100.0
            return side
        
        best_val = float('inf')
        best_params = None
        
        for init_angle in angles_to_try:
            if time.time() > deadline - 10:
                break
            
            init_params = [cx_init, cy_init, float(init_angle)]
            
            try:
                res = minimize(objective_outer, init_params, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1000})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
        
        if best_params is not None:
            cx, cy, angle = best_params
            side = compute_required_side(inner, (cx, cy), angle)
            if check_all_fit(inner, (cx, cy), side, angle) and check_all_disjoint(inner):
                return (cx, cy), angle, side
        return None

    def local_search(inner, ocx, ocy, oangle, seed):
        """Local search to refine inner hexagon positions."""
        nonlocal best_side, best_inner, best_outer_center, best_outer_angle, best_valid
        rng = random.Random(seed)
        cur_inner = [list(x) for x in inner]
        cur_S = compute_required_side(cur_inner, (ocx, ocy), oangle)
        cur_valid = check_all_fit(cur_inner, (ocx, ocy), cur_S, oangle) and check_all_disjoint(cur_inner)
        
        if not cur_valid:
            # Try to fix by moving hexagons
            for _ in range(30):
                idx = rng.randrange(N)
                old_x, old_y, old_a = cur_inner[idx]
                cur_inner[idx][0] = old_x + rng.gauss(0, 0.3)
                cur_inner[idx][1] = old_y + rng.gauss(0, 0.3)
                cur_S = compute_required_side(cur_inner, (ocx, ocy), oangle)
                cur_valid = check_all_fit(cur_inner, (ocx, ocy), cur_S, oangle) and check_all_disjoint(cur_inner)
                if cur_valid:
                    break
            if not cur_valid:
                return
        
        if cur_valid and cur_S < best_side:
            best_side = cur_S
            best_inner = [list(x) for x in cur_inner]
            best_outer_center = (ocx, ocy)
            best_outer_angle = oangle
            best_valid = True
        
        for it in range(500):
            if time.time() > deadline - 5:
                break
            old = [list(x) for x in cur_inner]
            idx = rng.randrange(N)
            cur_inner[idx][0] += rng.gauss(0, 0.05)
            cur_inner[idx][1] += rng.gauss(0, 0.05)
            
            new_S = compute_required_side(cur_inner, (ocx, ocy), oangle)
            new_valid = check_all_fit(cur_inner, (ocx, ocy), new_S, oangle) and check_all_disjoint(cur_inner)
            
            if new_valid and new_S < cur_S:
                cur_S = new_S
                if new_S < best_side:
                    best_side = new_S
                    best_inner = [list(x) for x in cur_inner]
                    best_outer_center = (ocx, ocy)
                    best_outer_angle = oangle
                    best_valid = True
            elif new_valid:
                if rng.random() < 0.05:
                    cur_S = new_S
            else:
                cur_inner = old

    # Try multiple lattice configurations with different spacings and angles
    base_spacing = SQRT3
    spacings = [base_spacing * (1 + i * 0.002) for i in range(8)]
    angles = [0, 15, 30, 45, 60, 75, 90]
    
    init_angles = [0, 30, 60, 90, 120, 150]
    
    for sp in spacings:
        for ang in angles:
            if time.time() > deadline - 20:
                break
            
            inner = generate_lattice_config(sp, ang)
            
            # Optimize outer hexagon
            result = optimize_outer(inner, 0.0, 0.0, init_angles)
            if result is not None:
                (ocx, ocy), oangle, S = result
                if S < best_side:
                    best_side = S
                    best_inner = [list(x) for x in inner]
                    best_outer_center = (ocx, ocy)
                    best_outer_angle = oangle
                    best_valid = True
                    # Run local search to refine
                    local_search(inner, ocx, ocy, oangle, int(sp * 1000) + ang * 10)

    # If we haven't found a valid solution, use a fallback
    if not best_valid or best_inner is None:
        # Simple linear arrangement as fallback
        sp = SQRT3 * 1.01
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_required_side(fallback_inner, (0.0, 0.0), 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_outer_center, best_side, best_outer_angle
# EVOLVE-BLOCK-END
