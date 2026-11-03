import math
import time
import numpy as np
import random
from scipy.optimize import minimize

N = 11

def _hex_vertices(cx, cy, angle, size=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle."""
    verts = []
    for k in range(6):
        a = angle + k * math.pi / 3
        verts.append((cx + size * math.cos(a), cy + size * math.sin(a)))
    return np.array(verts)

def _inner_hex(cx, cy, angle, size=1.0):
    """Return the 6 vertices of inner hexagon as array shape (6,2)."""
    return _hex_vertices(cx, cy, angle, size)

def _min_outer_side(points, outer_angle):
    """Compute minimum side length S of outer regular hexagon (centered at origin, rotated by outer_angle) containing all points."""
    # For a regular hexagon with side S centered at origin, rotated by outer_angle,
    # the containment condition: for each of 6 normals, max projection <= S*sqrt(3)/2
    # Actually, for a regular hexagon with "radius" (center to vertex) = S, 
    # the apothem (center to edge midpoint) = S*sqrt(3)/2
    # The hexagon with side length S has vertices at distance S from center.
    # The supporting lines are at distance S*sqrt(3)/2 from center.
    # So for normal direction n, all points must satisfy |p·n| <= S*sqrt(3)/2
    # Thus S >= 2/(sqrt(3)) * max_i max_k |p_i · n_k|
    
    k_angles = np.arange(6) * math.pi / 3 + outer_angle
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
    # points shape (n, 2), normals shape (6, 2)
    dots = points @ normals.T  # (n, 6)
    abs_dots = np.abs(dots)
    max_per_normal = np.max(abs_dots, axis=0)  # (6,)
    max_proj = np.max(max_per_normal)
    return (2.0 / math.sqrt(3)) * max_proj

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def _point_in_poly(point, poly):
    """Check if point is inside convex polygon (CCW)."""
    n = len(poly)
    for i in range(n):
        if _cross(poly[i], poly[(i+1) % n], point) < 0:
            return False
    return True

def _poly_overlap(v1, v2, eps=1e-9):
    """SAT check: do two convex polygons overlap? Returns True if they overlap (interior intersection)."""
    polys = [v1, v2]
    for v in polys:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                return False
    return True

def _in_outer(point, outer_center, outer_side, outer_angle):
    """Check if a point is inside the outer hexagon."""
    p = point - outer_center
    k_angles = np.arange(6) * math.pi / 3 + outer_angle
    normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
    dots = p @ normals.T
    apothem = outer_side * math.sqrt(3) / 2.0
    return np.all(np.abs(dots) <= apothem + 1e-9)

def _valid(inner_centers, inner_angles, outer_center, outer_side, outer_angle, eps=1e-9):
    """Check if packing is valid: no overlaps, all inside outer hexagon."""
    verts_list = [_inner_hex(inner_centers[i][0], inner_centers[i][1], inner_angles[i]) for i in range(N)]
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_overlap(verts_list[i], verts_list[j], eps):
                return False
    # Check containment
    for i in range(N):
        for v in verts_list[i]:
            if not _in_outer(v, outer_center, outer_side, outer_angle):
                return False
    return True

def _hexagonal_seed():
    """Create a hexagonal arrangement of 11 hexagons: 1 center + 6 ring + 4 more."""
    d = math.sqrt(3)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        centers.append((d * math.cos(angle), d * math.sin(angle)))
    # Add 4 more from ring 2
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    # Pick 4 from ring2 that are spread out
    random.seed(42)
    random.shuffle(ring2)
    centers.extend(ring2[:4])
    angles = [0.0] * N
    return centers, angles

def _hexagonal_seed2():
    """Alternative seed with slight rotation variations."""
    d = math.sqrt(3)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        centers.append((d * math.cos(angle), d * math.sin(angle)))
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append((x, y))
    random.seed(123)
    random.shuffle(ring2)
    centers.extend(ring2[:4])
    angles = [random.uniform(-0.1, 0.1) for _ in range(N)]
    return centers, angles

def _make_seed(variant=0):
    """Generate a seed packing configuration."""
    if variant == 0:
        return _hexagonal_seed()
    elif variant == 1:
        return _hexagonal_seed2()
    else:
        # Random perturbation of hexagonal seed
        centers, angles = _hexagonal_seed()
        random.seed(variant)
        centers = [(x + random.gauss(0, 0.1), y + random.gauss(0, 0.1)) for x, y in centers]
        angles = [a + random.gauss(0, 0.05) for a in angles]
        return centers, angles

def _try_refine(centers, angles, outer_angle, deadline, safety_margin=15.0):
    """Try to refine a packing using Nelder-Mead optimization."""
    start_time = time.time()
    
    def flatten(centers, angles):
        params = []
        for i in range(N):
            params.extend([centers[i][0], centers[i][1], angles[i]])
        return np.array(params)
    
    def unflatten(params):
        centers = [(params[3*i], params[3*i+1]) for i in range(N)]
        angles = [params[3*i+2] for i in range(N)]
        return centers, angles
    
    init_params = flatten(centers, angles)
    
    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1e10
        c, a = unflatten(params)
        # Compute S
        all_points = []
        for i in range(N):
            verts = _inner_hex(c[i][0], c[i][1], a[i])
            all_points.extend(verts.tolist())
        points_arr = np.array(all_points)
        S = _min_outer_side(points_arr, outer_angle)
        
        # Overlap penalty
        penalty = 0.0
        verts_list = [_inner_hex(c[i][0], c[i][1], a[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if _poly_overlap(verts_list[i], verts_list[j], 1e-9):
                    # Approximate penalty based on center distance
                    dx = c[i][0] - c[j][0]
                    dy = c[i][1] - c[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    penalty += max(0, math.sqrt(3) - dist) ** 2 * 50.0
        
        # Containment penalty
        for i in range(N):
            for v in verts_list[i]:
                if not _in_outer(v, (0,0), S, outer_angle):
                    penalty += 10.0
        
        return S + penalty
    
    best_params = init_params.copy()
    best_val = objective(init_params)
    
    try:
        res = minimize(objective, init_params, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6, 'initial_simplex': 0.1})
        
        if res.fun < best_val:
            best_params = res.x
            best_val = res.fun
    except Exception:
        pass
    
    c, a = unflatten(best_params)
    # Verify validity
    if _valid(c, a, (0,0), best_val, outer_angle):
        return c, a, best_val
    
    # If not valid, try to get S from a valid configuration
    all_points = []
    for i in range(N):
        verts = _inner_hex(c[i][0], c[i][1], a[i])
        all_points.extend(verts.tolist())
    points_arr = np.array(all_points)
    S = _min_outer_side(points_arr, outer_angle)
    
    if _valid(c, a, (0,0), S, outer_angle):
        return c, a, S
    
    return None

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 15.0
    
    best_result = None
    best_S = float('inf')
    
    # Try multiple seeds and outer angles
    outer_angles = [0.0, math.pi/6, math.pi/12, math.pi/3, math.pi/4]
    variants = range(6)
    
    for va in variants:
        if time.time() > deadline - safety_margin:
            break
        for oa in outer_angles:
            if time.time() > deadline - safety_margin:
                break
            centers, angles = _make_seed(va)
            result = _try_refine(centers, angles, oa, deadline, safety_margin)
            if result is not None:
                c, a, S = result
                if S < best_S:
                    best_S = S
                    best_result = (c, a, oa)
    
    # If we found a valid result, use it
    if best_result is not None:
        c, a, oa = best_result
        inner = [[c[i][0], c[i][1], math.degrees(a[i])] for i in range(N)]
        return inner, (0.0, 0.0), best_S, math.degrees(oa)
    
    # Fallback: straight line
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
