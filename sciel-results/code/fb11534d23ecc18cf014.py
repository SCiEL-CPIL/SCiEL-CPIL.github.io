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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
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

def _build_initial_conditions():
    """Build diverse initial conditions using axial coordinate lattice."""
    conditions = []
    
    # Axial coordinates for hexagonal lattice - center + ring1 + part of ring2
    # This gives a proper hexagonal packing structure
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
        return x, y
    
    # Condition 1: Standard hexagonal lattice, all aligned
    ic = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cartesian(q, r, 1.0)
        ic[3 * i] = x
        ic[3 * i + 1] = y
        ic[3 * i + 2] = 0.0
    ic[3 * N] = 0.0
    ic[3 * N + 1] = 0.0
    ic[3 * N + 2] = 0.0
    conditions.append(ic)
    
    # Condition 2: Slightly compressed lattice
    ic2 = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cartesian(q, r, 0.95)
        ic2[3 * i] = x
        ic2[3 * i + 1] = y
        ic2[3 * i + 2] = 0.0
    ic2[3 * N] = 0.0
    ic2[3 * N + 1] = 0.0
    ic2[3 * N + 2] = 0.0
    conditions.append(ic2)
    
    # Condition 3: Rotated inner hexagons by 30 degrees
    ic3 = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cartesian(q, r, 1.0)
        ic3[3 * i] = x
        ic3[3 * i + 1] = y
        ic3[3 * i + 2] = math.radians(30)
    ic3[3 * N] = 0.0
    ic3[3 * N + 1] = 0.0
    ic3[3 * N + 2] = 0.0
    conditions.append(ic3)
    
    # Condition 4: Different outer angle
    ic4 = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cartesian(q, r, 1.0)
        ic4[3 * i] = x
        ic4[3 * i + 1] = y
        ic4[3 * i + 2] = 0.0
    ic4[3 * N] = 0.0
    ic4[3 * N + 1] = 0.0
    ic4[3 * N + 2] = math.radians(30)
    conditions.append(ic4)
    
    # Condition 5: Perturbed lattice
    np.random.seed(42)
    ic5 = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cartesian(q, r, 1.0)
        ic5[3 * i] = x + np.random.randn() * 0.1
        ic5[3 * i + 1] = y + np.random.randn() * 0.1
        ic5[3 * i + 2] = np.random.randn() * 0.1
    ic5[3 * N] = np.random.randn() * 0.5
    ic5[3 * N + 1] = np.random.randn() * 0.5
    ic5[3 * N + 2] = 0.0
    conditions.append(ic5)
    
    # Condition 6: Scaled up lattice (looser)
    ic6 = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cartesian(q, r, 1.05)
        ic6[3 * i] = x
        ic6[3 * i + 1] = y
        ic6[3 * i + 2] = 0.0
    ic6[3 * N] = 0.0
    ic6[3 * N + 1] = 0.0
    ic6[3 * N + 2] = 0.0
    conditions.append(ic6)
    
    return conditions

def overlap_penalty(inner, strength=100.0):
    """Compute penalty for overlapping inner hexagons based on center distance."""
    penalty = 0.0
    n = len(inner)
    # Minimum center distance for non-overlapping unit hexagons
    min_d = 2.0 * SQRT3 - 0.2
    for i in range(n):
        cx1, cy1, _ = inner[i]
        for j in range(i + 1, n):
            cx2, cy2, _ = inner[j]
            d = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if d < min_d:
                excess = min_d - d
                penalty += strength * excess**2
    return penalty

def objective(params, deadline):
    """Objective function: smooth outer side length + overlap penalty."""
    if time.time() > deadline:
        return float('inf')
    
    n = N
    inner = []
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = math.degrees(params[3 * i + 2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3 * n]
    oy = params[3 * n + 1]
    outer_angle = math.degrees(params[3 * n + 2]) % 360
    
    S = compute_required_side_smooth(inner, (ox, oy), outer_angle)
    penalty = overlap_penalty(inner)
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    best_result = None
    best_score = float('inf')
    
    def try_result(params):
        nonlocal best_result, best_score
        inner = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            ang = math.degrees(params[3 * i + 2]) % 360
            inner.append([cx, cy, ang])
        ox = params[3 * N]
        oy = params[3 * N + 1]
        outer_angle = math.degrees(params[3 * N + 2]) % 360
        
        S = compute_required_side(inner, (ox, oy), outer_angle)
        
        if check_validity(inner, (ox, oy), S, outer_angle):
            if S < best_score:
                best_score = S
                best_result = (inner, (ox, oy), S, outer_angle)
    
    # Phase 1: L-BFGS-B refinement from multiple structural starting points
    initial_conditions = _build_initial_conditions()
    
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(objective, ic, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < float('inf'):
                try_result(res.x)
        except Exception:
            pass
    
    # Phase 2: Nelder-Mead refinement of outer parameters
    if best_result is not None and time.time() < deadline - 15:
        try:
            inner_fixed = best_result[0]
            
            def outer_obj(params_outer):
                ox, oy, oa = params_outer
                S = compute_required_side(inner_fixed, (ox, oy), math.degrees(oa) % 360)
                return S
            
            outer_ic = np.array([best_result[1][0], best_result[1][1], math.radians(best_result[3])])
            res_outer = minimize(outer_obj, outer_ic, method='Nelder-Mead',
                               options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_outer.fun < best_score:
                ox, oy, oa = res_outer.x
                outer_angle = math.degrees(oa) % 360
                S = res_outer.fun
                if check_validity(inner_fixed, (ox, oy), S, outer_angle):
                    best_result = (inner_fixed, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Phase 3: Final L-BFGS-B polish from best result
    if best_result is not None and time.time() < deadline - 5:
        try:
            params_best = np.zeros(3 * N + 3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3 * i] = cx
                params_best[3 * i + 1] = cy
                params_best[3 * i + 2] = math.radians(ang)
            params_best[3 * N] = best_result[1][0]
            params_best[3 * N + 1] = best_result[1][1]
            params_best[3 * N + 2] = math.radians(best_result[3])
            
            res = minimize(objective, params_best, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13})
            
            if res.fun < float('inf'):
                try_result(res.x)
        except Exception:
            pass
    
    # Fallback: hexagonal lattice
    if best_result is None:
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 1), (-2, 1), (-1, -1)
        ]
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        inner = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            inner.append([x, y, 0.0])
        S = compute_required_side(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result

# EVOLVE-BLOCK-END
