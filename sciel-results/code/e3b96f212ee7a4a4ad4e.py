import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_verts_vec(centers, angles_deg, side=1.0):
    """
    Vectorized computation of vertices for multiple hexagons.
    centers: (N, 2) array
    angles_deg: (N,) array
    Returns: (N, 6, 2) array
    """
    angles_rad = np.radians(angles_deg)
    # Angles of vertices relative to hexagon center
    # 0, 60, 120, 180, 240, 300 degrees
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    
    # (N, 1, 1) + (1, 6, 1)
    total_angles = angles_rad[:, None] + v_angles[None, :]
    
    cos_a = np.cos(total_angles)
    sin_a = np.sin(total_angles)
    
    # (N, 6, 1) * side
    dx = side * cos_a
    dy = side * sin_a
    
    # Add centers
    # centers is (N, 2) -> (N, 1, 2)
    verts = np.stack([dx + centers[:, 0:1], dy + centers[:, 1:2]], axis=-1)
    return verts

def min_outer_side_vec(inner_list, oc, oa_deg):
    """
    Vectorized minimum outer side calculation.
    inner_list: list of [x, y, angle_deg] or (N,3) array
    oc: (cx, cy)
    oa_deg: float
    """
    if isinstance(inner_list, list):
        inner_arr = np.array(inner_list)
    else:
        inner_arr = inner_list
        
    if inner_arr.size == 0:
        return 0.0

    centers = inner_arr[:, 0:2]
    angles = inner_arr[:, 2]
    
    verts = hex_verts_vec(centers, angles)
    
    # Flatten verts to (N*6, 2)
    verts_flat = verts.reshape(-1, 2)
    
    # Translate to outer center
    ox, oy = oc
    verts_rel = verts_flat - np.array([ox, oy])
    
    # Rotate to outer frame
    oa_rad = math.radians(oa_deg)
    cos_a = math.cos(-oa_rad)
    sin_a = math.sin(-oa_rad)
    
    # Rotation matrix R = [[cos, -sin], [sin, cos]]
    # v' = R v
    verts_rot = np.zeros_like(verts_rel)
    verts_rot[:, 0] = verts_rel[:, 0] * cos_a - verts_rel[:, 1] * sin_a
    verts_rot[:, 1] = verts_rel[:, 0] * sin_a + verts_rel[:, 1] * cos_a
    
    # Normals of outer hexagon
    # 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = np.stack([np.cos(normal_angles), np.sin(normal_angles)], axis=1)
    
    # Project all vertices onto all normals
    # verts_rot: (M, 2), normals: (6, 2)
    # projs: (M, 6)
    projs = verts_rot @ normals.T
    
    # Max projection
    max_proj = np.max(projs)
    
    # Side = 2 * Apothem / sqrt(3)
    return 2.0 * max_proj / SQRT3

def overlap_penalty_vec(inner_list, min_dist=SQRT3 - 0.001):
    """
    Vectorized overlap penalty based on center distance.
    """
    if isinstance(inner_list, list):
        inner_arr = np.array(inner_list)
    else:
        inner_arr = inner_list
        
    n = inner_arr.shape[0]
    if n < 2:
        return 0.0
        
    xs = inner_arr[:, 0]
    ys = inner_arr[:, 1]
    
    # Pairwise distances
    # Using broadcasting
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy
    
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])
    
    violations = np.maximum(0.0, min_dist - dists)
    return float(np.sum(violations * violations))

def check_disjoint(inner_list, eps=1e-7):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    # This is slow, so we only call it on best candidates
    n = len(inner_list)
    for i in range(n):
        vi = hex_verts(inner_list[i][0], inner_list[i][1], 1.0, math.radians(inner_list[i][2]))
        for j in range(i + 1, n):
            vj = hex_verts(inner_list[j][0], inner_list[j][1], 1.0, math.radians(inner_list[j][2]))
            if sat_overlap(vi, vj, eps):
                return False
    return True

def hex_verts(cx, cy, s, ang_rad):
    """Compute vertices of a regular hexagon."""
    v = np.zeros((6, 2))
    for i in range(6):
        theta = ang_rad + i * math.pi / 3.0
        v[i, 0] = cx + s * math.cos(theta)
        v[i, 1] = cy + s * math.sin(theta)
    return v

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

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Leave 10s margin for safety
    
    # 1. Initialize with a compact hexagonal lattice sorted by distance
    # Generate axial coordinates for rings 0, 1, 2
    coords = []
    coords.append((0,0))
    # Ring 1
    for q in range(-1, 2):
        for r in range(-1, 2):
            if q == 0 and r == 0: continue
            # Check if in ring 1: max(abs(q), abs(r), abs(-q-r)) == 1
            if max(abs(q), abs(r), abs(-q-r)) == 1:
                coords.append((q,r))
    # Ring 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            if q == 0 and r == 0: continue
            if max(abs(q), abs(r), abs(-q-r)) == 2:
                coords.append((q,r))
                
    # Sort by distance from origin (approx Cartesian distance)
    def dist_key(c):
        q, r = c
        return 3 * (q*q + q*r + r*r)
        
    coords.sort(key=dist_key)
    
    # Take first N
    axial_coords = coords[:N]
    
    # Convert to Cartesian
    scale = 1.0 + 1e-4 
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial inner layout: [x, y, angle]
    initial_inner = np.array([[x, y, 0.0] for x, y in centers])
    
    # Initial outer center and angle
    cx_avg = np.mean(initial_inner[:, 0])
    cy_avg = np.mean(initial_inner[:, 1])
    init_outer_center = (cx_avg, cy_avg)
    init_outer_angle = 0.0
    
    best_side = float('inf')
    best_inner = initial_inner.copy()
    best_outer_center = init_outer_center
    best_outer_angle = init_outer_angle
    
    # Helper to update best
    def update_best(inner, oc, oa):
        nonlocal best_side, best_inner, best_outer_center, best_outer_angle
        s = min_outer_side_vec(inner, oc, oa)
        if s < best_side and check_disjoint(inner):
            best_side = s
            best_inner = inner.copy()
            best_outer_center = (oc[0], oc[1])
            best_outer_angle = oa
            
    update_best(initial_inner, init_outer_center, init_outer_angle)
    
    # 2. Phase 1: Optimize Outer Container with Fixed Inner Layout
    # This finds the best "bounding box" for the initial layout.
    def obj_outer(params):
        cx, cy, ang = params
        return min_outer_side_vec(initial_inner, (cx, cy), ang)
        
    best_outer_params = None
    best_outer_val = float('inf')
    
    # Try a few initial angles to avoid local minima
    init_angles = [0, 30, 60, 90, 15, 45, 75, -15, -45, -75]
    
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
        
    # 3. Phase 2: Joint Optimization via Powell (Deterministic Local Search)
    # Powell is often better than SA for this dimensionality if initialized well.
    
    def joint_objective(params):
        # params: [x1, y1, a1, ..., xN, yN, aN, ox, oy, oa]
        inner_list = params[:N*3].reshape(N, 3)
        ocx, ocy, oang = params[N*3], params[N*3+1], params[N*3+2]
        
        side = min_outer_side_vec(inner_list, (ocx, ocy), oang)
        pen = overlap_penalty_vec(inner_list)
        return side + 1000.0 * pen
        
    # Initialize parameters from best outer fit
    init_params = np.concatenate([initial_inner.flatten(), [best_outer_center[0], best_outer_center[1], best_outer_angle]])
    
    # Try Powell from a few starting points
    # 1. The current best
    # 2. Perturbed versions
    
    starts = [
        init_params,
        init_params + np.random.normal(0, 0.1, init_params.shape),
        init_params + np.random.normal(0, 0.5, init_params.shape),
    ]
    
    # Add some random restarts if time permits
    if time.time() < deadline - 40:
        for _ in range(5):
            random_inner = initial_inner + np.random.normal(0, 1.0, initial_inner.shape)
            random_params = np.concatenate([random_inner.flatten(), [best_outer_center[0], best_outer_center[1], best_outer_angle]])
            starts.append(random_params)

    for s in starts:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(joint_objective, s, method='Powell', 
                           options={'maxiter': 10000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            # Check validity of the result
            res_inner = res.x[:N*3].reshape(N, 3)
            res_oc = (res.x[N*3], res.x[N*3+1])
            res_oa = res.x[N*3+2]
            
            if check_disjoint(res_inner):
                res_side = min_outer_side_vec(res_inner, res_oc, res_oa)
                if res_side < best_side:
                    best_side = res_side
                    best_inner = res_inner
                    best_outer_center = res_oc
                    best_outer_angle = res_oa
        except Exception:
            pass
            
    # 4. Phase 3: Final High-Precision Polish on Outer Container
    # Fix inner layout to best_inner, optimize outer only.
    def obj_outer_polish(params):
        cx, cy, ang = params
        return min_outer_side_vec(best_inner, (cx, cy), ang)
        
    best_polish_val = best_side
    best_polish_params = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Use Powell for better precision on small dimensions
    starts_polish = [
        best_polish_params,
        best_polish_params + np.array([0.01, 0, 0]),
        best_polish_params + np.array([0, 0.01, 0]),
        best_polish_params + np.array([0, 0, 1.0]),
        best_polish_params + np.array([0, 0, -1.0])
    ]
    
    for s in starts_polish:
        if time.time() > deadline - 2:
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
        
    return best_inner.tolist(), best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
