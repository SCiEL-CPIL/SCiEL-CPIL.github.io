"""Refined program for hexagon packing; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_verts(cx, cy, s, ang_rad):
    """Compute vertices of a regular hexagon."""
    v = np.zeros((6, 2))
    for i in range(6):
        theta = ang_rad + i * math.pi / 3.0
        v[i, 0] = cx + s * math.cos(theta)
        v[i, 1] = cy + s * math.sin(theta)
    return v

def min_outer_side(inner_list, oc, oa_deg):
    """
    Calculate the minimum side length of the outer hexagon to contain all inner hexagons.
    inner_list: list of [x, y, angle_deg]
    oc: tuple (cx, cy)
    oa_deg: float, angle of outer hexagon
    """
    if not inner_list:
        return 0.0
    
    ox, oy = oc
    oa_rad = math.radians(oa_deg)
    
    # Normals of the outer hexagon relative to its center, in the outer's frame
    # Outward normals of a unit hexagon (side 1) are at 30, 90, 150, 210, 270, 330 degrees
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_proj = 0.0
    
    # Transformation from global to outer-local frame
    # Rotate by -oa_rad
    cos_a = math.cos(-oa_rad)
    sin_a = math.sin(-oa_rad)
    
    for ix, iy, ia_deg in inner_list:
        ia_rad = math.radians(ia_deg)
        # Vertices of this inner hexagon in global coords
        verts = hex_verts(ix, iy, 1.0, ia_rad)
        
        for vx, vy in verts:
            # Translate to outer center
            dx = vx - ox
            dy = vy - oy
            
            # Rotate to outer frame
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            
            # Project onto all 6 normals
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_proj:
                    max_proj = proj
    
    # The apothem of the outer hexagon must be >= max_proj
    # Side = 2 * Apothem / sqrt(3)
    return 2.0 * max_proj / SQRT3

def min_outer_side_smooth(inner_list, oc, oa_deg, beta=80.0):
    """
    Smooth version of min_outer_side using log-sum-exp for the max operation.
    Useful for gradient-based methods, but we mostly use derivative-free.
    """
    if not inner_list:
        return 0.0
    
    ox, oy = oc
    oa_rad = math.radians(oa_deg)
    
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
        
    cos_a = math.cos(-oa_rad)
    sin_a = math.sin(-oa_rad)
    
    projs = []
    for ix, iy, ia_deg in inner_list:
        ia_rad = math.radians(ia_deg)
        verts = hex_verts(ix, iy, 1.0, ia_rad)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for nx, ny in normals:
                projs.append(lx * nx + ly * ny)
                
    if not projs:
        return 0.0
        
    projs_arr = np.array(projs)
    # Soft max
    max_proj = float(np.log(np.sum(np.exp(beta * projs_arr))) / beta)
    
    return 2.0 * max_proj / SQRT3

def sat_overlap(p1, p2, eps=1e-9):
    """
    Separating Axis Theorem check for two convex polygons.
    Returns True if they overlap.
    """
    axes = []
    for i in range(len(p1)):
        p_a = p1[i]
        p_b = p1[(i + 1) % len(p1)]
        edge = (p_b[0] - p_a[0], p_b[1] - p_a[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
            
    for i in range(len(p2)):
        p_a = p2[i]
        p_b = p2[(i + 1) % len(p2)]
        edge = (p_b[0] - p_a[0], p_b[1] - p_a[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
            
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in p1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in p1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in p2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in p2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner_list, eps=1e-7):
    """Check if all inner hexagons are pairwise disjoint."""
    n = len(inner_list)
    for i in range(n):
        vi = hex_verts(inner_list[i][0], inner_list[i][1], 1.0, math.radians(inner_list[i][2]))
        for j in range(i + 1, n):
            vj = hex_verts(inner_list[j][0], inner_list[j][1], 1.0, math.radians(inner_list[j][2]))
            if sat_overlap(vi, vj, eps):
                return False
    return True

def overlap_penalty(inner_list, min_dist=SQRT3 - 0.001):
    """
    Penalty for overlapping hexagons.
    Uses center distance as a proxy. For unit hexagons, min center distance is sqrt(3).
    """
    n = len(inner_list)
    if n < 2:
        return 0.0
        
    xs = np.array([p[0] for p in inner_list])
    ys = np.array([p[1] for p in inner_list])
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    # Penalty is quadratic in the violation amount
    violations = np.maximum(0.0, min_dist - dists)
    return float(np.sum(violations * violations))

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin for safety
    
    # 1. Initialize with a compact hexagonal lattice
    # 11 hexagons: Center + 6 ring 1 + 4 from ring 2
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    
    # Ensure we have exactly N
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
        
    # Convert to Cartesian
    # Using a slight expansion to ensure they don't start touching, allowing SA to settle
    scale = 1.0 + 1e-4 
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial inner layout: [x, y, angle]
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    # Initial outer center and angle
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    init_outer_center = (cx_avg, cy_avg)
    init_outer_angle = 0.0
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = init_outer_center
    best_outer_angle = init_outer_angle
    
    # Helper to update best
    def update_best(inner, oc, oa):
        nonlocal best_side, best_inner, best_outer_center, best_outer_angle
        s = min_outer_side(inner, oc, oa)
        if s < best_side and check_disjoint(inner):
            best_side = s
            best_inner = [list(i) for i in inner]
            best_outer_center = (oc[0], oc[1])
            best_outer_angle = oa
    
    update_best(initial_inner, init_outer_center, init_outer_angle)
    
    # 2. Phase 1: Optimize Outer Container with Fixed Inner Layout
    # This finds the best "bounding box" for the initial layout.
    def obj_outer(params):
        cx, cy, ang = params
        return min_outer_side(initial_inner, (cx, cy), ang)
        
    best_outer_params = None
    best_outer_val = float('inf')
    
    # Try a few initial angles to avoid local minima
    init_angles = [0, 30, 60, 90, 15, 45, 75]
    
    for ang in init_angles:
        if time.time() > deadline - 60:
            break
            
        x0 = [cx_avg, cy_avg, float(ang)]
        try:
            res = minimize(obj_outer, x0, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_outer_val:
                best_outer_val = res.fun
                best_outer_params = res.x
        except Exception:
            pass
            
    if best_outer_params is not None:
        best_outer_center = (best_outer_params[0], best_outer_params[1])
        best_outer_angle = best_outer_params[2]
        best_side = best_outer_val
        
    # 3. Phase 2: Joint Optimization via Simulated Annealing
    # Optimize inner positions and outer container simultaneously.
    
    def joint_objective(params):
        inner_list = []
        for i in range(N):
            inner_list.append([params[i*3], params[i*3+1], params[i*3+2]])
        ocx, ocy, oang = params[N*3], params[N*3+1], params[N*3+2]
        
        side = min_outer_side(inner_list, (ocx, ocy), oang)
        pen = overlap_penalty(inner_list)
        return side + 1000.0 * pen
    
    # Initialize parameters from best outer fit
    init_params = []
    for x, y, a in initial_inner:
        init_params.extend([x, y, a])
    init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    current_params = np.array(init_params)
    current_val = joint_objective(current_params)
    best_params = current_params.copy()
    best_val_joint = current_val
    
    # Update best if valid
    cur_inner = [[current_params[i*3], current_params[i*3+1], current_params[i*3+2]] for i in range(N)]
    cur_oc = (current_params[N*3], current_params[N*3+1])
    cur_oa = current_params[N*3+2]
    if check_disjoint(cur_inner):
        s = min_outer_side(cur_inner, cur_oc, cur_oa)
        if s < best_side:
            best_side = s
            best_inner = cur_inner
            best_outer_center = cur_oc
            best_outer_angle = cur_oa
    
    T = 0.5
    T_min = 0.0005
    cooling = 0.9999
    
    # Estimate time budget
    elapsed = time.time() - start_time
    remaining = deadline - 20 - elapsed
    if remaining < 5:
        remaining = 5
        
    # Time a single evaluation
    t0 = time.time()
    _ = joint_objective(current_params)
    eval_time = max(time.time() - t0, 1e-6)
    
    max_iter = int(min(remaining / eval_time, 50000))
    
    for i in range(max_iter):
        if time.time() > deadline - 20:
            break
            
        neighbor = current_params.copy()
        r = np.random.random()
        
        if r < 0.8:
            # Move an inner hexagon
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 3)
            if dim == 2: # Angle
                delta = np.random.normal(0, 1.5)
            else:
                delta = np.random.normal(0, 0.1)
            neighbor[idx*3 + dim] += delta
        else:
            # Move outer container
            dim = np.random.randint(N*3, N*3+3)
            if dim == N*3+2: # Angle
                delta = np.random.normal(0, 1.5)
            else:
                delta = np.random.normal(0, 0.05)
            neighbor[dim] += delta
            
        neighbor_val = joint_objective(neighbor)
        
        # Accept if better, or with probability based on temperature
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
        else:
            delta_e = neighbor_val - current_val
            if delta_e < 0 or np.random.random() < math.exp(-delta_e / T):
                current_params = neighbor
                current_val = neighbor_val
                
        if current_val < best_val_joint:
            best_params = current_params.copy()
            best_val_joint = current_val
            
            # Check validity and update global best
            b_inner = [[best_params[i*3], best_params[i*3+1], best_params[i*3+2]] for i in range(N)]
            b_oc = (best_params[N*3], best_params[N*3+1])
            b_oa = best_params[N*3+2]
            if check_disjoint(b_inner):
                b_side = min_outer_side(b_inner, b_oc, b_oa)
                if b_side < best_side:
                    best_side = b_side
                    best_inner = b_inner
                    best_outer_center = b_oc
                    best_outer_angle = b_oa
        
        T *= cooling
        if T < T_min:
            break
            
    # 4. Phase 3: Final High-Precision Polish on Outer Container
    # Fix inner layout to best_inner, optimize outer only.
    def obj_outer_polish(params):
        cx, cy, ang = params
        return min_outer_side(best_inner, (cx, cy), ang)
        
    best_polish_val = best_side
    best_polish_params = [best_outer_center[0], best_outer_center[1], best_outer_angle]
    
    # Use Powell for better precision on small dimensions
    starts = [
        best_polish_params,
        [best_outer_center[0] + 0.01, best_outer_center[1], best_outer_angle],
        [best_outer_center[0], best_outer_center[1] + 0.01, best_outer_angle],
        [best_outer_center[0], best_outer_center[1], best_outer_angle + 1.0],
        [best_outer_center[0], best_outer_center[1], best_outer_angle - 1.0]
    ]
    
    for s in starts:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(obj_outer_polish, s, method='Powell', 
                           options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14})
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
