"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# Precompute hexagon vertex offsets for unit hexagon at angle 0
_HEX_ANGLES = np.pi / 3.0 * np.arange(6)
_HEX_COS = np.cos(_HEX_ANGLES)
_HEX_SIN = np.sin(_HEX_ANGLES)
# Outer hexagon support normals (inward-pointing)
_OUTER_NORMALS = np.array([[math.cos(math.radians(30 + i*60)), math.sin(math.radians(30 + i*60))] for i in range(6)])

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + _HEX_ANGLES
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

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
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        ang = math.radians(iangle)
        ca, sa = math.cos(ang), math.sin(ang)
        # Vertices of unit hexagon
        for k in range(6):
            vx = _HEX_COS[k]
            vy = _HEX_SIN[k]
            # Rotate vertex
            rvx = vx * ca - vy * sa
            rvy = vx * sa + vy * ca
            # Translate
            wx = ix + rvx
            wy = iy + rvy
            # Transform to outer frame
            dx = wx - ox
            dy = wy - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            # Project onto all 6 normals
            for m in range(6):
                proj = lx * _OUTER_NORMALS[m, 0] + ly * _OUTER_NORMALS[m, 1]
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=100.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    projections = np.empty(N * 6 * 6)
    idx = 0
    for (ix, iy, iangle) in inner:
        ang = math.radians(iangle)
        ca, sa = math.cos(ang), math.sin(ang)
        for k in range(6):
            vx = _HEX_COS[k]
            vy = _HEX_SIN[k]
            rvx = vx * ca - vy * sa
            rvy = vx * sa + vy * ca
            wx = ix + rvx
            wy = iy + rvy
            dx = wx - ox
            dy = wy - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for m in range(6):
                projections[idx] = lx * _OUTER_NORMALS[m, 0] + ly * _OUTER_NORMALS[m, 1]
                idx += 1
    
    max_proj = np.max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = beta * (projections - max_proj)
    sum_exp = np.sum(np.exp(scaled))
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
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
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts_list.append(get_hex_vertices(ix, iy, 1.0, math.radians(iangle)))
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_list[i], verts_list[j], eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
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

def objective(params, use_smooth=True, beta=100.0):
    """Objective function: outer side length + penalty for overlaps."""
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    if use_smooth:
        S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta)
    else:
        S = compute_required_side(inner, (ox, oy), outer_angle)
    
    penalty = 0.0
    verts_list = []
    for (cx, cy, ang) in inner:
        verts_list.append(get_hex_vertices(cx, cy, 1.0, math.radians(ang)))
    
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, _ = inner[i]
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                if sat_overlap_check(verts_list[i], verts_list[j], eps=1e-8):
                    penalty += (2.0 - d) ** 2 * 100
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Axial coordinates for 11 hexagons in a compact cluster
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    # Generate diverse initial conditions
    initial_conditions = []
    
    # Base positions
    base_centers = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
    
    def make_ic(angles, scale=1.0, outer_angle=0.0, ox=0.0, oy=0.0):
        ic = np.zeros(3 * N + 3)
        for i, (px, py) in enumerate(base_centers):
            ic[3*i] = px * scale
            ic[3*i+1] = py * scale
            ic[3*i+2] = np.radians(angles[i])
        ic[3*N] = ox
        ic[3*N+1] = oy
        ic[3*N+2] = np.radians(outer_angle)
        return ic
    
    # 1. Standard lattice, all angles 0
    initial_conditions.append(make_ic([0.0]*N))
    
    # 2. Outer ring rotated 30 deg
    # Indices: 0 is center. 1-6 are ring 1. 7-10 are ring 2.
    # We can rotate all non-center hexagons.
    angles_rot30 = [0.0] + [30.0]*(N-1)
    initial_conditions.append(make_ic(angles_rot30))
    
    # 3. Outer ring rotated 15 deg
    angles_rot15 = [0.0] + [15.0]*(N-1)
    initial_conditions.append(make_ic(angles_rot15))
    
    # 4. Random angles
    np.random.seed(42)
    angles_rand = [np.random.uniform(0, 180) for _ in range(N)]
    initial_conditions.append(make_ic(angles_rand))
    
    # 5. Rotated outer hexagon 30 deg
    initial_conditions.append(make_ic([0.0]*N, outer_angle=30.0))
    
    # 6. Rotated outer hexagon 15 deg
    initial_conditions.append(make_ic([0.0]*N, outer_angle=15.0))
    
    # 7. Perturbed standard
    ic7 = make_ic([0.0]*N)
    ic7 += np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic7)
    
    # 8. Perturbed rotated
    ic8 = make_ic(angles_rot30)
    ic8 += np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic8)
    
    best_result = None
    best_score = float('inf')
    
    # Optimize each initial condition
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 15:
            break
        
        try:
            # First pass: smooth objective
            res = minimize(lambda p: objective(p, use_smooth=True, beta=50.0), ic, 
                          method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            # Second pass: refine with exact objective
            if res.fun < best_score or best_result is None:
                res2 = minimize(lambda p: objective(p, use_smooth=False), res.x, 
                               method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if res2.fun < best_score:
                    params = res2.x
                    inner = []
                    for i in range(N):
                        cx = params[3*i]
                        cy = params[3*i+1]
                        ang = np.degrees(params[3*i+2]) % 360
                        inner.append((cx, cy, ang))
                    ox = params[3*N]
                    oy = params[3*N+1]
                    outer_angle = np.degrees(params[3*N+2]) % 360
                    
                    S = compute_required_side(inner, (ox, oy), outer_angle)
                    
                    if check_validity(inner, (ox, oy), S, outer_angle):
                        if S < best_score:
                            best_score = S
                            best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        centers_fb = [axial_to_cartesian(q, r, 1.0) for q, r in axial_coords]
        inner = [(px, py, 0.0) for px, py in centers_fb]
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Refine if time permits
    if time.time() < deadline - 20:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(lambda p: objective(p, use_smooth=False), params_best, 
                          method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                params = res.x
                inner = []
                for i in range(N):
                    cx = params[3*i]
                    cy = params[3*i+1]
                    ang = np.degrees(params[3*i+2]) % 360
                    inner.append((cx, cy, ang))
                ox = params[3*N]
                oy = params[3*N+1]
                outer_angle = np.degrees(params[3*N+2]) % 360
                S = compute_required_side(inner, (ox, oy), outer_angle)
                
                if check_validity(inner, (ox, oy), S, outer_angle):
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
            
    return best_result

# EVOLVE-BLOCK-END
