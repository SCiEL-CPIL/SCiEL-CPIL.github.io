"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize
from itertools import combinations

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

def hex_vertices_local(angle_rad):
    """Return 6 vertices of a unit hexagon centered at origin."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        verts.append((math.cos(theta), math.sin(theta)))
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
    Smooth surrogate for compute_required_side using log-sum-exp.
    Returns a smooth approximation of the max projection.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    # Collect all projections
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    all_projs = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                all_projs.append(proj)
    
    if not all_projs:
        return 0.0
    
    # Log-sum-exp for smooth max
    arr = np.array(all_projs)
    m = np.max(arr)
    if m < -1000:
        return 0.0
    log_sum = m + np.log(np.sum(np.exp((arr - m) * beta)) / beta)
    # Smooth max ≈ log_sum / beta when beta is large
    smooth_max = log_sum / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons overlap using SAT.
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
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def build_lattice_config(axial_coords, scale=1.0):
    """Convert axial coordinates to Cartesian centers with given scale."""
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
    return centers

def rigid_composite_objective(params, centers, inner_angles_rad):
    """
    Objective for rigid composite: all hexagons move/rotate together.
    params: [cx, cy, angle_deg] - outer hexagon center and angle
    The inner hexagons are rigidly placed relative to each other.
    """
    cx, cy, angle_deg = params
    return compute_required_side_smooth(
        [[centers[i][0], centers[i][1], math.degrees(inner_angles_rad[i])] for i in range(len(centers))],
        (cx, cy), angle_deg
    )

def semi_rigid_objective(params, base_centers, inner_angles_rad):
    """
    Semi-rigid: outer hexagon is fixed, individual inner angles can vary.
    params: [angle_0, angle_1, ..., angle_{N-1}] in degrees
    """
    inner = [[base_centers[i][0], base_centers[i][1], params[i]] for i in range(len(params))]
    return compute_required_side_smooth(inner, (0, 0), 0.0)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Axial coordinate pool
    axial_pool = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            if max(abs(q), abs(r), abs(q+r)) <= 3:
                axial_pool.append((q, r))

    # Known compact arrangements
    original_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    alt_coords_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (2, -1), (0, 2)
    ]
    
    alt_coords_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-1, 2), (1, 1), (0, 2)
    ]

    dist2_points = [(q, r) for q, r in axial_pool if max(abs(q), abs(r), abs(q+r)) == 2]

    import random
    random.seed(42)

    candidate_arrangements = [original_coords, alt_coords_1, alt_coords_2]
    
    for _ in range(20):
        if time.time() > deadline - 100:
            break
        if len(dist2_points) >= 4:
            outer4 = random.sample(dist2_points, 4)
            candidate = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)] + outer4
            valid = all(p in axial_pool for p in candidate)
            if valid and len(set(candidate)) == N:
                candidate_arrangements.append(candidate)

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_centers = None
    best_inner_angles_rad = None

    # Phase 1: Rigid composite optimization for each candidate
    for axial_coords in candidate_arrangements:
        if time.time() > deadline - 60:
            break
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        epsilon = 1e-5
        scale = 1.0 + epsilon
        centers = build_lattice_config(axial_coords, scale)
        inner_angles_rad = [0.0] * len(centers)
        
        # Compute centroid for initialization
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        def objective_rigid(params):
            return rigid_composite_objective(params, centers, inner_angles_rad)
        
        best_val = float('inf')
        best_result = None
        
        init_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
        
        for init_angle in init_angles:
            if time.time() > deadline - 50:
                break
            init_params = [cx_avg, cy_avg, float(init_angle)]
            try:
                res = minimize(objective_rigid, init_params, method='Nelder-Mead', 
                              options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 8000})
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res.x
            except Exception:
                pass
                
        if best_result is not None:
            # Use exact (non-smooth) evaluation for final comparison
            inner_for_eval = [[centers[i][0], centers[i][1], 0.0] for i in range(len(centers))]
            exact_side = compute_required_side(inner_for_eval, (best_result[0], best_result[1]), best_result[2])
            if exact_side < best_side:
                best_side = exact_side
                best_inner = inner_for_eval
                best_outer_center = (best_result[0], best_result[1])
                best_outer_angle = best_result[2]
                best_centers = centers
                best_inner_angles_rad = inner_angles_rad[:]

    if best_inner is None:
        axial_coords = original_coords[:N]
        epsilon = 1e-5
        scale = 1.0 + epsilon
        centers = build_lattice_config(axial_coords, scale)
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_centers = centers
        best_inner_angles_rad = [0.0] * N
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # Phase 2: Semi-rigid optimization - vary individual inner angles
    if time.time() < deadline - 30 and best_centers is not None:
        n_hex = len(best_centers)
        base_centers = best_centers
        
        def objective_semi_rigid(params):
            return semi_rigid_objective(params, base_centers, [0.0]*n_hex)
        
        init_params = [0.0] * n_hex
        
        try:
            res = minimize(objective_semi_rigid, init_params, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 15000})
            
            # Check disjointness with new angles
            new_inner = [[base_centers[i][0], base_centers[i][1], res.x[i]] for i in range(n_hex)]
            if check_all_disjoint(new_inner, eps=1e-5):
                # Evaluate with the best outer container
                new_side = compute_required_side(new_inner, best_outer_center, best_outer_angle)
                if new_side < best_side:
                    best_side = new_side
                    best_inner = new_inner
                    best_inner_angles_rad = [math.radians(res.x[i]) for i in range(n_hex)]
        except Exception:
            pass

    # Phase 3: Final high-precision polish on outer container
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    if time.time() < deadline - 3:
        try:
            res = minimize(objective_outer_polish, best_polish_params, method='Nelder-Mead',
                          options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 40000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
    
    if best_polish_params is not None:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val
    
    # Final validation
    if not check_all_disjoint(best_inner, eps=1e-6):
        # Fallback to guaranteed valid configuration
        axial_coords = original_coords[:N]
        epsilon = 1e-5
        scale = 1.0 + epsilon
        centers = build_lattice_config(axial_coords, scale)
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
