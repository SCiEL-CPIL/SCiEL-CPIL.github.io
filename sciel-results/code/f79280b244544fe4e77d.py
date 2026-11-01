"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

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
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
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
    
    proj_arr = np.array(projections)
    max_proj = float(np.max(proj_arr))
    # Numerically stable log-sum-exp
    shifted = beta * (proj_arr - max_proj)
    max_proj_smooth = max_proj + math.log(np.sum(np.exp(shifted))) / beta
    
    return 2.0 * max_proj_smooth / SQRT3

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

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Compute a smooth penalty for overlaps based on center distances."""
    n = len(inner)
    if n < 2:
        return 0.0
    
    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initial layout: 11 hexagons in a tight cluster
    # Center (0,0), Ring 1 (6), Ring 2 (4 partial)
    # Using axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # We will optimize the scale of this initial layout to be as tight as possible
    # But first, let's just place them at unit scale
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
    
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (cx_avg, cy_avg)
    best_outer_angle = 0.0
    
    # Phase 1: Optimize outer container with fixed inner layout
    # We assume the inner layout is "good enough" for now, just find the best container
    
    def objective_outer_smooth(params, beta=20.0):
        cx, cy, angle_deg = params
        return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=beta)
    
    def objective_outer_exact(params):
        cx, cy, angle_deg = params
        return compute_required_side(initial_inner, (cx, cy), angle_deg)
    
    best_result = None
    best_val = float('inf')
    
    # Coarse angle sweep
    init_angles = [0, 30, 60, 90, 120, 150]
    
    for init_angle in init_angles:
        if time.time() > deadline - 80:
            break
        
        init_params = [cx_avg, cy_avg, float(init_angle)]
        
        try:
            # Smooth phase
            res = minimize(lambda p: objective_outer_smooth(p, 20.0), init_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500})
            
            # Exact phase
            res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 1000})
            
            if res2.fun < best_val:
                best_val = res2.fun
                best_result = res2.x
        except Exception:
            pass
    
    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    
    # Phase 2: Joint optimization
    # We use Differential Evolution for global search
    # The objective is: Container Size + Penalty * Overlap
    
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
        
        # Use smooth version for DE to avoid getting stuck in local minima of the max function
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta=30.0)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    # Bounds
    # Inner hexagons should stay within a reasonable range around the center
    bounds = []
    for i in range(N):
        bounds.append((-5, 5))
        bounds.append((-5, 5))
        bounds.append((0, 60)) # Angle symmetry
    
    # Outer center and angle
    bounds.append((cx_avg - 2, cx_avg + 2))
    bounds.append((cy_avg - 2, cy_avg + 2))
    bounds.append((0, 60))
    
    initial_params = []
    for x, y, a in initial_inner:
        initial_params.extend([x, y, a])
    initial_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Run DE
    de_success = False
    best_params = np.array(initial_params)
    
    try:
        # popsize=20, maxiter=50
        # Time limit check inside DE is not direct, so we estimate
        # Each eval is ~0.5ms, 20*50*33 params = 33000 evals ~ 16s
        result_de = differential_evolution(
            objective_joint, 
            bounds, 
            seed=42, 
            maxiter=50, 
            popsize=20, 
            tol=1e-6,
            polish=False # We will polish later
        )
        
        if result_de.fun < best_side or (not check_all_disjoint(initial_inner)):
            # Check if DE result is valid
            de_inner = []
            for i in range(N):
                de_inner.append([result_de.x[i*3], result_de.x[i*3+1], result_de.x[i*3+2]])
            
            de_ocx = result_de.x[N*3]
            de_ocy = result_de.x[N*3+1]
            de_oangle = result_de.x[N*3+2]
            
            if check_all_disjoint(de_inner, eps=1e-5):
                de_side = compute_required_side(de_inner, (de_ocx, de_ocy), de_oangle)
                if de_side < best_side:
                    best_side = de_side
                    best_inner = de_inner
                    best_outer_center = (de_ocx, de_ocy)
                    best_outer_angle = de_oangle
                    best_params = result_de.x
                    de_success = True
                else:
                    # Even if not better, if current best is invalid, take this
                    if not check_all_disjoint(best_inner):
                        best_side = de_side
                        best_inner = de_inner
                        best_outer_center = (de_ocx, de_ocy)
                        best_outer_angle = de_oangle
                        best_params = result_de.x
                        de_success = True
    except Exception:
        pass
    
    # If DE didn't improve or failed, try a local search from the best known valid point
    # We use L-BFGS-B with a smooth objective to refine
    
    def objective_refine(params):
        inner_list = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            inner_list.append([x, y, a])
        
        ocx = params[N*3]
        ocy = params[N*3+1]
        oangle = params[N*3+2]
        
        side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta=50.0)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 1000.0 * penalty
    
    # Try refining from the current best
    if time.time() < deadline - 30:
        try:
            res_refine = minimize(
                objective_refine, 
                best_params, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            
            refine_inner = []
            for i in range(N):
                refine_inner.append([res_refine.x[i*3], res_refine.x[i*3+1], res_refine.x[i*3+2]])
            
            refine_ocx = res_refine.x[N*3]
            refine_ocy = res_refine.x[N*3+1]
            refine_oangle = res_refine.x[N*3+2]
            
            if check_all_disjoint(refine_inner, eps=1e-5):
                refine_side = compute_required_side(refine_inner, (refine_ocx, refine_ocy), refine_oangle)
                if refine_side < best_side:
                    best_side = refine_side
                    best_inner = refine_inner
                    best_outer_center = (refine_ocx, refine_ocy)
                    best_outer_angle = refine_oangle
                    best_params = res_refine.x
        except Exception:
            pass
    
    # Phase 3: Final high-precision polish on outer container
    # Fix inner, optimize outer exactly
    
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    polish_starts = [
        [best_outer_center[0], best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 1],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 1],
    ]
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    for ps in polish_starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
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
