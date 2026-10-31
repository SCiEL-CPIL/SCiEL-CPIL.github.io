import math
import time
import numpy as np
from scipy.optimize import minimize

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

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, k=100.0):
    """
    Compute a smooth approximation of the minimum outer side length.
    Uses log-sum-exp for the max projection.
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
    projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                # We need max absolute projection?
                # The hexagon contains points where |proj| <= apothem.
                # So we need max(|proj|).
                # Let's collect all |proj| values.
                projs.append(abs(proj))

    if not projs:
        return 0.0

    # Soft max of projs
    projs = np.array(projs)
    max_proj = np.max(projs)
    # Avoid overflow
    if max_proj > 0:
        # log-sum-exp trick
        log_sum_exp = max_proj + np.log(np.sum(np.exp(projs - max_proj)))
        soft_max = log_sum_exp / k
    else:
        soft_max = 0.0
        
    # The actual side length is 2 * max_proj / SQRT3
    # So soft side length is 2 * soft_max / SQRT3
    return 2.0 * soft_max / SQRT3

def compute_exact_side(inner, outer_center, outer_angle_deg):
    """
    Compute the exact minimum outer side length.
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
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = abs(lx * nx + ly * ny)
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3

def hex_gap(a, b):
    """
    Compute the minimum gap between two unit hexagons.
    Returns positive if separated, negative if overlapping.
    """
    va = get_hex_vertices(a[0], a[1], 1.0, math.radians(a[2]))
    vb = get_hex_vertices(b[0], b[1], 1.0, math.radians(b[2]))
    
    # Get all edge normals
    axes = []
    for poly in [va, vb]:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
            if norm > 1e-9:
                axes.append((normal[0] / norm, normal[1] / norm))
    
    min_gap = float('inf')
    for axis in axes:
        min_a = min(p[0] * axis[0] + p[1] * axis[1] for p in va)
        max_a = max(p[0] * axis[0] + p[1] * axis[1] for p in va)
        min_b = min(p[0] * axis[0] + p[1] * axis[1] for p in vb)
        max_b = max(p[0] * axis[0] + p[1] * axis[1] for p in vb)
        
        gap = max(min_a - max_b, min_b - max_a)
        if gap > min_gap:
            min_gap = gap
    
    return -min_gap

def smooth_overlap_penalty(inner, k=50.0):
    """
    Compute a smooth penalty for overlaps.
    """
    penalty = 0.0
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            gap = hex_gap(inner[i], inner[j])
            # If gap < 0, they overlap.
            # Use softplus to smooth the max(0, -gap)
            # softplus(x) = log(1 + exp(x))
            # We want penalty for negative gap.
            # Let x = -gap. If gap is -1 (overlap 1), x=1.
            # We want a large penalty.
            # Use x^2 if x > 0, else 0. Smoothed:
            # softplus(x)^2 is not quite right for depth.
            # Let's use: if gap < 0, penalty = gap^2.
            # To smooth: 
            # g_smooth = (log(1+exp(k*gap)) - log(2)) / k  # This is soft-min of 0 and gap?
            # Actually, just use gap^2 * sigmoid(k*gap)?
            
            # Simpler: 
            # Penalty = (softplus(-k*gap))^2 / k^2 ?
            # If gap = -1, -k*gap = k. softplus(k) ~ k. Penalty ~ 1.
            # If gap = 1, -k*gap = -k. softplus(-k) ~ 0. Penalty ~ 0.
            
            x = -k * gap
            # Avoid overflow
            if x > 50:
                sp = x
            elif x < -50:
                sp = 0.0
            else:
                sp = math.log(1 + math.exp(x))
            
            penalty += (sp / k) ** 2
    return penalty

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    verts_cache = []
    for i in range(n):
        x, y, ang = inner[i]
        verts = get_hex_vertices(x, y, 1.0, math.radians(ang))
        verts_cache.append(verts)
    
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(verts_cache[i], verts_cache[j], eps):
                return False
    return True

def sat_overlap_check(v1, v2, eps=1e-7):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    """
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
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

def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if abs(lx * nx + ly * ny) > apothem + eps:
                    return False
    return True

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    # Initial configuration: hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    epsilon = 1e-6
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # --- Phase 1: Quick outer hexagon optimization ---
    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        return compute_exact_side(inner_list, (cx, cy), angle_deg)

    best_outer_val = float('inf')
    best_outer_params = [cx_avg, cy_avg, 0.0]

    for init_angle in [0, 30, 60, 90, 120, 150]:
        if time.time() - start_time > DEADLINE - 60:
            break

        try:
            res = minimize(
                lambda p: objective_outer(p, initial_inner),
                [cx_avg, cy_avg, float(init_angle)],
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 500}
            )
            if res.fun < best_outer_val:
                best_outer_val = res.fun
                best_outer_params = res.x.copy()
        except Exception:
            pass

    if best_outer_val < float('inf'):
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        best_side = best_outer_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_exact_side(initial_inner, best_outer_center, best_outer_angle)

    # --- Phase 2: Multi-start Optimization with Smooth Penalty ---
    
    def objective_joint_smooth(params):
        if time.time() - start_time > DEADLINE - 5:
            return 1e10

        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        # Smooth side length
        side = compute_required_side_smooth(inner_list, (cx, cy), angle, k=50.0)
        
        # Smooth overlap penalty
        penalty = smooth_overlap_penalty(inner_list, k=50.0)
        
        return side + 10.0 * penalty

    # Run multiple optimizations from different starts
    starts = [
        initial_inner,
    ]
    
    # Add perturbed starts
    for _ in range(5):
        perturbed = []
        for x, y, a in initial_inner:
            perturbed.append([x + rng.normal(0, 0.5), y + rng.normal(0, 0.5), a + rng.normal(0, 30)])
        starts.append(perturbed)
    
    for start_inner in starts:
        if time.time() - start_time > DEADLINE - 20:
            break
        
        init_params = []
        for x, y, a in start_inner:
            init_params.extend([x, y, a])
        init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

        try:
            res = minimize(
                objective_joint_smooth,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000}
            )
            
            new_inner = [[res.x[i*3], res.x[i*3+1], res.x[i*3+2]] for i in range(N)]
            new_cx, new_cy, new_angle = res.x[N*3], res.x[N*3+1], res.x[N*3+2]
            
            # Validate
            if check_all_disjoint(new_inner):
                new_side = compute_exact_side(new_inner, (new_cx, new_cy), new_angle)
                if check_inside_outer(new_inner, (new_cx, new_cy), new_side, new_angle):
                    if new_side < best_side:
                        best_side = new_side
                        best_inner = new_inner
                        best_outer_center = (new_cx, new_cy)
                        best_outer_angle = new_angle
        except Exception:
            pass

    # --- Phase 3: Final Polish ---
    if time.time() - start_time < DEADLINE - 10 and best_side < float('inf'):
        
        def objective_joint_exact(params):
            if time.time() - start_time > DEADLINE - 2:
                return 1e10

            inner_list = []
            for i in range(N):
                x = params[i * 3]
                y = params[i * 3 + 1]
                a = params[i * 3 + 2]
                inner_list.append([x, y, a])

            cx = params[N * 3]
            cy = params[N * 3 + 1]
            angle = params[N * 3 + 2]

            side = compute_exact_side(inner_list, (cx, cy), angle)

            # Penalty for overlaps
            penalty = 0.0
            for i in range(N):
                for j in range(i + 1, N):
                    d = hex_gap(inner_list[i], inner_list[j])
                    if d < 0:
                        penalty += (-d) ** 2 * 100.0

            return side + penalty

        init_params_joint = []
        for x, y, a in best_inner:
            init_params_joint.extend([x, y, a])
        init_params_joint.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

        try:
            res_joint = minimize(
                objective_joint_exact,
                init_params_joint,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 2000}
            )

            new_inner = [[res_joint.x[i*3], res_joint.x[i*3+1], res_joint.x[i*3+2]] for i in range(N)]
            new_cx, new_cy, new_angle = res_joint.x[N*3], res_joint.x[N*3+1], res_joint.x[N*3+2]

            if check_all_disjoint(new_inner):
                new_side = compute_exact_side(new_inner, (new_cx, new_cy), new_angle)
                if check_inside_outer(new_inner, (new_cx, new_cy), new_side, new_angle) and new_side < best_side:
                    best_side = new_side
                    best_inner = new_inner
                    best_outer_center = (new_cx, new_cy)
                    best_outer_angle = new_angle
        except Exception:
            pass

    # Final safety check
    if best_side == float('inf'):
        best_inner = initial_inner
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_exact_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle
