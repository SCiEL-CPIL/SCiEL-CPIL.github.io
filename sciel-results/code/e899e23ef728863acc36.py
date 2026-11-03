"""
Packing 12 unit hexagons into a regular hexagon.
Strategy:
1. Define hexagon geometry and support functions.
2. Implement SAT-based collision detection.
3. Implement objective function: outer side length + collision penalty.
4. Use Simulated Annealing to find a good configuration.
5. Refine with local search.
"""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 148.0  # Safety margin below 150s

# Precompute hexagon vertex coordinates for unit side length
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
HEX_VERTS = np.array([
    [math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))] for k in range(6)
])

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a hexagon centered at (cx, cy) with given angle and side."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    # Rotation matrix
    R = np.array([[ca, -sa], [sa, ca]])
    verts = HEX_VERTS * side
    verts = verts @ R.T
    return verts + np.array([cx, cy])

def _hex_support(cx, cy, angle_deg, nx, ny, side=1.0):
    """Max of dot product of hex vertices with direction (nx,ny)."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    max_dot = 0.0
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        rx = ca * x - sa * y
        ry = sa * x + ca * y
        d = nx * rx + ny * ry
        if d > max_dot:
            max_dot = d
    return max_dot + nx * cx + ny * cy

def _hex_min_support(cx, cy, angle_deg, nx, ny, side=1.0):
    """Min of dot product of hex vertices with direction (nx,ny)."""
    return -_hex_support(cx, cy, angle_deg, -nx, -ny, side)

def _outer_hex_side(cx_arr, cy_arr, angle_arr, ocx, ocy, oangle_deg):
    """
    Compute the minimum side length of an outer regular hexagon 
    centered at (ocx, ocy) with angle oangle_deg that encloses all inner hexagons.
    """
    apothem_unit = math.sqrt(3) / 2.0
    oa = math.radians(oangle_deg)
    max_need = 0.0
    # The 6 normals of the outer hexagon sides
    for k in range(6):
        na = oa + math.radians(90 + 60 * k)
        nx, ny = math.cos(na), math.sin(na)
        for i in range(N):
            s = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], nx, ny)
            # Distance from outer center to the support plane
            val = (s - nx * ocx - ny * ocy) / apothem_unit
            if val > max_need:
                max_need = val
    return max_need

def _is_valid(cx_arr, cy_arr, angle_arr, tol=1e-9):
    """
    Check if all hexagons are disjoint using SAT.
    Returns True if valid, False otherwise.
    """
    n = len(cx_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist_sq = dx * dx + dy * dy
            # Quick reject: if distance > sum of circumradii, they might still collide if rotated,
            # but if distance is very large, skip.
            # Max distance for collision: 2 * circumradius = 2.0
            # Actually, max projection is 1.0 (side) + 1.0 (side) = 2.0? No.
            # Circumradius R = 1.0. Max distance between centers for contact is 2.0.
            # If dist > 2.0, they definitely don't touch.
            if dist_sq > 4.0:
                continue
            
            # SAT test: check all 12 edge normals (6 from each hex)
            for ai in range(6):
                aia = math.radians(angle_arr[i] + 90 + 60 * ai)
                anx, any_ = math.cos(aia), math.sin(aia)
                minA = _hex_min_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                maxA = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                minB = _hex_min_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                maxB = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                if maxA <= minB + tol or maxB <= minA + tol:
                    break
            else:
                for aj in range(6):
                    aja = math.radians(angle_arr[j] + 90 + 60 * aj)
                    anx, any_ = math.cos(aja), math.sin(aja)
                    minA = _hex_min_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                    maxA = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], anx, any_)
                    minB = _hex_min_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                    maxB = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], anx, any_)
                    if maxA <= minB + tol or maxB <= minA + tol:
                        break
                else:
                    return False
    return True

def _collision_penalty(cx_arr, cy_arr, angle_arr, eps=0.01):
    """
    Smooth penalty for collisions.
    Uses approximate distance based on support functions along the connecting line.
    """
    penalty = 0.0
    n = len(cx_arr)
    for i in range(n):
        for j in range(i + 1, n):
            dx = cx_arr[i] - cx_arr[j]
            dy = cy_arr[i] - cy_arr[j]
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                return 1e6
            nx, ny = dx / dist, dy / dist
            # Support of i in direction n
            sup_i = _hex_support(cx_arr[i], cy_arr[i], angle_arr[i], nx, ny)
            # Support of j in direction -n
            sup_j = _hex_support(cx_arr[j], cy_arr[j], angle_arr[j], -nx, -ny)
            
            # Projected distances from centers
            d_i = sup_i - (nx * cx_arr[i] + ny * cy_arr[i])
            d_j = sup_j - (-nx * cx_arr[j] - ny * cy_arr[j])
            
            # Separation along the line
            sep = dist - d_i - d_j
            if sep < eps:
                penalty += (eps - sep) ** 2
    return penalty

def _objective(params, ocx, ocy, oangle_deg, beta=100.0):
    """
    Objective function for optimization.
    params: [x0, y0, a0, x1, y1, a1, ...]
    """
    cx = params[0::3]
    cy = params[1::3]
    ang = params[2::3]
    
    side = _outer_hex_side(cx, cy, ang, ocx, ocy, oangle_deg)
    pen = _collision_penalty(cx, cy, ang, eps=0.05)
    
    return side + beta * pen

def _make_initial_configs():
    """Generate diverse initial configurations."""
    configs = []
    
    # 1. Flower pattern: 1 center, 6 ring, 5 more
    pts = [[0.0, 0.0, 0.0]]
    r1 = 2 * math.sqrt(3)
    for i in range(6):
        angle = math.radians(60 * i)
        pts.append([r1 * math.cos(angle), r1 * math.sin(angle), 0.0])
    r2 = r1 * 1.732
    for deg in [30, 90, 150, 210, 270]:
        rad = math.radians(deg)
        pts.append([r2 * math.cos(rad), r2 * math.sin(rad), 0.0])
    configs.append(pts[:12])
    
    # 2. Lattice pattern
    pts = []
    rh = math.sqrt(3)
    cw = 2.0
    # Row 0
    for c in range(4):
        pts.append([c * cw - 1.5 * cw, -rh, 0.0])
    # Row 1
    for c in range(4):
        pts.append([c * cw - 1.0 * cw, 0.0, 0.0])
    # Row 2
    for c in range(4):
        pts.append([c * cw - 1.5 * cw, rh, 0.0])
    configs.append(pts[:12])
    
    # 3. Random
    for _ in range(5):
        pts = []
        for _ in range(N):
            pts.append([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(0, 60)])
        configs.append(pts)
        
    return configs

def construct():
    t0 = time.time()
    random.seed(42)
    
    best_side = float('inf')
    best_cfg = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    configs = _make_initial_configs()
    
    # Evaluate initial configs
    for cfg in configs:
        if time.time() - t0 > TIME_LIMIT - 60:
            break
        if len(cfg) != N:
            continue
        cx_arr = np.array([p[0] for p in cfg])
        cy_arr = np.array([p[1] for p in cfg])
        angle_arr = np.array([p[2] for p in cfg])
        
        if _is_valid(cx_arr, cy_arr, angle_arr):
            for oangle in [0, 30, 60, 90, 120, 150]:
                side = _outer_hex_side(cx_arr, cy_arr, angle_arr, 0.0, 0.0, float(oangle))
                if side < best_side:
                    best_side = side
                    best_cfg = [row[:] for row in cfg]
                    best_center = (0.0, 0.0)
                    best_angle = float(oangle)
    
    if best_cfg is None:
        best_cfg = configs[0]
        cx_arr = np.array([p[0] for p in best_cfg])
        cy_arr = np.array([p[1] for p in best_cfg])
        angle_arr = np.array([p[2] for p in best_cfg])
        best_side = _outer_hex_side(cx_arr, cy_arr, angle_arr, 0.0, 0.0, 0.0)
    
    # Simulated Annealing
    # Parameters
    T = 1.0
    T_min = 1e-5
    alpha = 0.999
    iters_per_temp = 50
    
    current_params = np.zeros(3 * N)
    for i in range(N):
        current_params[3*i] = best_cfg[i][0]
        current_params[3*i+1] = best_cfg[i][1]
        current_params[3*i+2] = best_cfg[i][2]
        
    current_cost = _objective(current_params, best_center[0], best_center[1], best_angle)
    
    best_params = current_params.copy()
    best_cost = current_cost
    
    step_size = 0.5
    
    while T > T_min and time.time() - t0 < TIME_LIMIT - 10:
        for _ in range(iters_per_temp):
            if time.time() - t0 > TIME_LIMIT - 10:
                break
            
            # Generate neighbor
            new_params = current_params.copy()
            idx = random.randint(0, 3*N - 1)
            if idx % 3 == 2: # Angle
                new_params[idx] += random.gauss(0, step_size)
            else: # Position
                new_params[idx] += random.gauss(0, step_size)
                
            new_cost = _objective(new_params, best_center[0], best_center[1], best_angle)
            
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_params = new_params
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_params = current_params.copy()
            
        T *= alpha
        
    # Final refinement: Optimize outer hexagon for the best inner config
    cx_opt = best_params[0::3]
    cy_opt = best_params[1::3]
    ang_opt = best_params[2::3]
    
    if _is_valid(cx_opt, cy_opt, ang_opt):
        best_local_side = float('inf')
        # Fine-tune outer hexagon
        for oangle in np.arange(0, 180, 5):
            for ocx in np.arange(-1, 1, 0.5):
                for ocy in np.arange(-1, 1, 0.5):
                    s = _outer_hex_side(cx_opt, cy_opt, ang_opt, ocx, ocy, float(oangle))
                    if s < best_local_side:
                        best_local_side = s
                        best_center = (float(ocx), float(ocy))
                        best_angle = float(oangle)
        
        if best_local_side < best_side:
            best_side = best_local_side
            best_cfg = [[float(cx_opt[i]), float(cy_opt[i]), float(ang_opt[i])] for i in range(N)]
    else:
        # If invalid, try to fix by pushing apart or use previous best
        # For now, stick to the previous best valid config if this one is invalid
        pass

    # Ensure we return a valid config
    if best_cfg is None or not _is_valid(np.array([p[0] for p in best_cfg]), np.array([p[1] for p in best_cfg]), np.array([p[2] for p in best_cfg])):
         # Fallback to first valid initial config or generate a simple one
         # We already checked initial configs, so best_cfg should be valid if any were valid
         if not _is_valid(np.array([p[0] for p in best_cfg]), np.array([p[1] for p in best_cfg]), np.array([p[2] for p in best_cfg])):
             # Last resort: scatter them far apart
             best_cfg = []
             for i in range(N):
                 best_cfg.append([i * 3.0, 0.0, 0.0])
             best_center = (0.0, 0.0)
             best_angle = 0.0
             best_side = _outer_hex_side(np.array([p[0] for p in best_cfg]), np.array([p[1] for p in best_cfg]), np.array([p[2] for p in best_cfg]), 0.0, 0.0, 0.0)

    return best_cfg, best_center, best_side, best_angle
