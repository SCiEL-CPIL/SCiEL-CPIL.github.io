"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def convex_hull(points):
    """Compute convex hull of a set of 2D points using Andrew's monotone chain."""
    points = np.array(points)
    points = np.unique(points, axis=0)
    if len(points) <= 2:
        return points
    points = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    return np.vstack([lower[:-1], upper[:-1]])

def support_function(vertices, normals):
    """Compute support function values: max dot product of each vertex with each normal."""
    return np.max(vertices @ normals.T, axis=1)

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    # The 6 outward normals of the outer hexagon
    normals = np.array([[np.cos(oa + np.radians(30 + 60 * k)), np.sin(oa + np.radians(30 + 60 * k))]
                        for k in range(6)])
    
    # Collect all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Translate relative to outer center
    all_verts_rel = all_verts - np.array([ox, oy])
    
    # For each normal, compute the max support value
    all_normals = np.vstack([normals, -normals])
    supports = np.max(all_verts_rel @ all_normals.T, axis=1)
    
    min_S = np.max(supports) / (np.sqrt(3.0) / 2.0)
    return max(min_S, 0.0)

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i + 1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i + 1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            if polygons_overlap(v1, v2, tol=1e-7):
                return False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    n_outer = len(outer_verts)
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            inside = True
            for i in range(n_outer):
                v1 = outer_verts[i]
                v2 = outer_verts[(i + 1) % n_outer]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i + 2) % n_outer] - v1)
                if d1 * d2 < -1e-9:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def overlap_penalty(inner_hexagons, strength=500.0):
    """Compute a smooth penalty for overlapping inner hexagons."""
    penalty = 0.0
    n = len(inner_hexagons)
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            d = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            # Minimum center distance for non-overlapping unit hexagons (any orientation)
            min_d = 2.0 * np.sqrt(3.0) - 0.1
            if d < min_d:
                excess = min_d - d
                penalty += strength * excess ** 2
    return penalty

def objective(params, deadline):
    """Objective function: outer side length + penalty for overlaps."""
    if time.time() > deadline:
        return float('inf')
    
    n = N
    inner = []
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        ang = np.degrees(params[3 * i + 2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3 * n]
    oy = params[3 * n + 1]
    outer_angle = np.degrees(params[3 * n + 2]) % 360
    
    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
    penalty = overlap_penalty(inner)
    
    return S + penalty

def _build_initial_conditions():
    """Build diverse initial conditions for optimization."""
    conditions = []
    
    # Hexagonal lattice: center + ring of 6 + 4 from ring 2
    hex_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
        if len(hex_positions) >= N:
            break
    hex_positions = hex_positions[:N]
    
    # Condition 1: Standard hexagonal lattice
    ic = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(hex_positions):
        ic[3 * i] = px
        ic[3 * i + 1] = py
        ic[3 * i + 2] = 0.0
    ic[3 * N] = 0.0
    ic[3 * N + 1] = 0.0
    ic[3 * N + 2] = 0.0
    conditions.append(ic)
    
    # Condition 2: Perturbed lattice
    np.random.seed(123)
    ic2 = ic + np.random.randn(3 * N + 3) * 0.15
    conditions.append(ic2)
    
    # Condition 3: Different outer angle
    ic3 = ic.copy()
    ic3[3 * N + 2] = np.radians(30)
    conditions.append(ic3)
    
    # Condition 4: Rotated inner hexagons
    ic4 = ic.copy()
    for i in range(N):
        ic4[3 * i + 2] = np.radians(30)
    conditions.append(ic4)
    
    # Condition 5: Compressed lattice
    ic5 = ic.copy()
    for i in range(N):
        ic5[3 * i] *= 0.92
        ic5[3 * i + 1] *= 0.92
    conditions.append(ic5)
    
    # Condition 6: Random scatter
    np.random.seed(456)
    ic6 = np.random.randn(3 * N + 3) * 2.0
    ic6[3 * N + 2] = 0.0
    conditions.append(ic6)
    
    # Condition 7: Linear arrangement
    ic7 = np.zeros(3 * N + 3)
    for i in range(N):
        ic7[3 * i] = (i - N // 2) * 2 * np.sqrt(3)
        ic7[3 * i + 1] = 0.0
    conditions.append(ic7)
    
    # Discrete lattice search: try multiple lattice spacings and rotations
    lattice_spacings = [1.8, 2.0, 2.2, 2.4, 2.6]
    lattice_rotations = [0, 15, 30, 45, 60]
    
    for spacing in lattice_spacings:
        for rot_deg in lattice_rotations:
            rot = np.radians(rot_deg)
            # Generate hexagonal lattice points within a radius
            points = []
            max_r = spacing * 3
            for m in range(-3, 4):
                for n in range(-3, 4):
                    # Hexagonal lattice coordinates
                    x = spacing * (m + 0.5 * n)
                    y = spacing * (np.sqrt(3) / 2 * n)
                    r = np.sqrt(x**2 + y**2)
                    if r <= max_r:
                        # Apply rotation
                        xr = x * np.cos(rot) - y * np.sin(rot)
                        yr = x * np.sin(rot) + y * np.cos(rot)
                        points.append((xr, yr))
            
            # Sort by distance from origin and take N closest
            points.sort(key=lambda p: p[0]**2 + p[1]**2)
            if len(points) >= N:
                points = points[:N]
            else:
                continue
            
            ic_lat = np.zeros(3 * N + 3)
            for i, (px, py) in enumerate(points):
                ic_lat[3 * i] = px
                ic_lat[3 * i + 1] = py
                ic_lat[3 * i + 2] = 0.0
            ic_lat[3 * N + 2] = rot
            conditions.append(ic_lat)
    
    return conditions

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = float('inf')
    
    def try_result(params):
        nonlocal best_result, best_score
        inner = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            ang = np.degrees(params[3 * i + 2]) % 360
            inner.append([cx, cy, ang])
        ox = params[3 * N]
        oy = params[3 * N + 1]
        outer_angle = np.degrees(params[3 * N + 2]) % 360
        
        S = compute_outer_side_length(inner, (ox, oy), outer_angle)
        
        if check_validity(inner, (ox, oy), S, outer_angle):
            if S < best_score:
                best_score = S
                best_result = (inner, (ox, oy), S, outer_angle)
    
    # Phase 1: Discrete lattice search - explore globally distinct structural arrangements
    if time.time() < deadline - 80:
        initial_conditions = _build_initial_conditions()
        
        for ic in initial_conditions:
            if time.time() > deadline - 80:
                break
            try:
                # Quick evaluation without full optimization
                try_result(ic)
            except Exception:
                pass
    
    # Phase 2: Differential Evolution for global exploration
    if time.time() < deadline - 60:
        try:
            # Bounds for each parameter
            bounds = []
            for i in range(N):
                bounds.append((-8, 8))   # x
                bounds.append((-8, 8))   # y
                bounds.append((0, 2 * np.pi))  # angle
            bounds.append((-5, 5))       # outer cx
            bounds.append((-5, 5))       # outer cy
            bounds.append((0, 2 * np.pi))  # outer angle
            
            de_result = differential_evolution(
                objective, bounds, args=(deadline,),
                maxiter=50, popsize=15, tol=1e-6,
                mutation=(0.3, 1.0), recombination=0.9,
                seed=42, polish=False,
                update_interval=5,
                callback=lambda x, convergence: (time.time() > deadline)
            )
            
            if de_result.fun < float('inf'):
                try_result(de_result.x)
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B refinement from multiple starting points
    initial_conditions = _build_initial_conditions()
    
    # Add DE result if available
    if best_result is not None:
        ic_de = np.zeros(3 * N + 3)
        for i, (cx, cy, ang) in enumerate(best_result[0]):
            ic_de[3 * i] = cx
            ic_de[3 * i + 1] = cy
            ic_de[3 * i + 2] = np.radians(ang)
        ic_de[3 * N] = best_result[1][0]
        ic_de[3 * N + 1] = best_result[1][1]
        ic_de[3 * N + 2] = np.radians(best_result[3])
        initial_conditions.append(ic_de)
    
    for ic in initial_conditions:
        if time.time() > deadline - 40:
            break
        
        try:
            res = minimize(objective, ic, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.fun < float('inf'):
                try_result(res.x)
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead refinement of outer parameters
    if best_result is not None and time.time() < deadline - 20:
        try:
            inner_fixed = best_result[0]
            
            def outer_obj(params_outer):
                ox, oy, oa = params_outer
                S = compute_outer_side_length(inner_fixed, (ox, oy), oa)
                return S
            
            outer_ic = np.array([best_result[1][0], best_result[1][1], np.radians(best_result[3])])
            res_outer = minimize(outer_obj, outer_ic, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_outer.fun < best_score:
                ox, oy, oa = res_outer.x
                outer_angle = np.degrees(oa) % 360
                S = res_outer.fun
                if check_validity(inner_fixed, (ox, oy), S, outer_angle):
                    best_result = (inner_fixed, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Phase 5: Final L-BFGS-B polish
    if best_result is not None and time.time() < deadline - 10:
        try:
            params_best = np.zeros(3 * N + 3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3 * i] = cx
                params_best[3 * i + 1] = cy
                params_best[3 * i + 2] = np.radians(ang)
            params_best[3 * N] = best_result[1][0]
            params_best[3 * N + 1] = best_result[1][1]
            params_best[3 * N + 2] = np.radians(best_result[3])
            
            res = minimize(objective, params_best, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < float('inf'):
                try_result(res.x)
        except Exception:
            pass
    
    # Fallback: hexagonal lattice
    if best_result is None:
        hex_positions = [(0.0, 0.0)]
        for k in range(6):
            angle = np.radians(60 * k)
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            hex_positions.append((x, y))
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2 * np.sqrt(3) * np.cos(angle)
            y = 2 * np.sqrt(3) * np.sin(angle)
            hex_positions.append((x, y))
            if len(hex_positions) >= N:
                break
        hex_positions = hex_positions[:N]
        inner = [[px, py, 0.0] for px, py in hex_positions]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result

# EVOLVE-BLOCK-END
