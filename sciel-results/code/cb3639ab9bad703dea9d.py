"""Combined program for hexagon packing."""
import math
import time
import numpy as np
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

def hexagon_distance(c1, a1, c2, a2):
    """
    Compute the minimum distance between two unit hexagons with centers c1, c2
    and angles a1, a2 (in radians). Returns 0 if overlapping.
    Uses support function approach.
    """
    v1 = get_hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = get_hex_vertices(c2[0], c2[1], 1.0, a2)
    
    # Collect all edge normals from both hexagons
    axes = []
    for verts in [v1, v2]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            norm_len = math.sqrt(edge[0]**2 + edge[1]**2)
            if norm_len > 1e-12:
                nx = -edge[1] / norm_len
                ny = edge[0] / norm_len
                axes.append((nx, ny))
    
    min_gap = float('inf')
    for nx, ny in axes:
        min1 = min(v[0] * nx + v[1] * ny for v in v1)
        max1 = max(v[0] * nx + v[1] * ny for v in v1)
        min2 = min(v[0] * nx + v[1] * ny for v in v2)
        max2 = max(v[0] * nx + v[1] * ny for v in v2)
        
        # Gap on this axis (positive means separated)
        gap = max(min2 - max1, min1 - max2)
        if gap < min_gap:
            min_gap = gap
    
    # If min_gap > 0, polygons are separated by at least min_gap
    # If min_gap <= 0, they overlap or touch
    return max(0.0, min_gap)

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """
    Compute a smooth penalty for overlaps based on actual hexagon distances.
    """
    n = len(inner)
    if n < 2:
        return 0.0
    
    penalty = 0.0
    for i in range(n):
        xi, yi, ai = inner[i]
        ai_rad = math.radians(ai)
        for j in range(i + 1, n):
            xj, yj, aj = inner[j]
            aj_rad = math.radians(aj)
            
            dist = hexagon_distance((xi, yi), ai_rad, (xj, yj), aj_rad)
            if dist < min_dist:
                diff = min_dist - dist
                penalty += diff * diff
    
    return float(penalty)

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            # Simple SAT check
            axes = []
            for verts in [vi, vj]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k + 1) % 6]
                    edge = (p2[0] - p1[0], p2[1] - p1[1])
                    norm = math.sqrt(edge[0]**2 + edge[1]**2)
                    if norm > 1e-9:
                        axes.append((-edge[1]/norm, edge[0]/norm))
            
            for ax, ay in axes:
                min1 = min(p[0]*ax + p[1]*ay for p in vi)
                max1 = max(p[0]*ax + p[1]*ay for p in vi)
                min2 = min(p[0]*ax + p[1]*ay for p in vj)
                max2 = max(p[0]*ax + p[1]*ay for p in vj)
                
                if max1 < min2 + eps or max2 < min1 + eps:
                    break
                else:
                    # Check if this axis shows separation
                    pass
            
            # Re-implementing SAT properly for clarity in check
            overlap = True
            for ax, ay in axes:
                min1 = min(p[0]*ax + p[1]*ay for p in vi)
                max1 = max(p[0]*ax + p[1]*ay for p in vi)
                min2 = min(p[0]*ax + p[1]*ay for p in vj)
                max2 = max(p[0]*ax + p[1]*ay for p in vj)
                
                if max1 < min2 + eps or max2 < min1 + eps:
                    overlap = False
                    break
            
            if overlap:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Initial Layout: Hexagonal Cluster
    # Center + 6 neighbors + 4 more from second ring
    # Axial coords (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (0, 2)
    ]
    
    # Convert to Cartesian
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    scale = 1.0
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    # Calculate centroid to center the configuration
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Shift centers to be around 0,0 for better numerical conditioning
    for i in range(N):
        initial_inner[i][0] -= cx_avg
        initial_inner[i][1] -= cy_avg

    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 2. Simulated Annealing on Joint Space
    # Params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, oangle]
    # Total 36 params
    
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
        
        side = compute_required_side(inner_list, (ocx, ocy), oangle)
        penalty = compute_overlap_penalty(inner_list)
        
        return side + 100.0 * penalty

    current_params = np.array([v for row in initial_inner for v in row] + [0.0, 0.0, 0.0])
    current_val = objective_joint(current_params)
    best_params = current_params.copy()
    best_val = current_val

    # SA Parameters
    T = 0.5
    T_min = 0.001
    cooling_rate = 0.995
    max_iter = 5000
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if time.time() > deadline - 20:
            break
            
        # Generate neighbor
        neighbor = current_params.copy()
        # Perturb random subset of variables
        num_perturb = 3
        indices = rng.choice(len(neighbor), num_perturb, replace=False)
        for idx in indices:
            neighbor[idx] += rng.normal(0, 0.1)
            
        neighbor_val = objective_joint(neighbor)
        
        if neighbor_val < current_val or rng.random() < math.exp((current_val - neighbor_val) / T):
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val:
                best_val = current_val
                best_params = current_params.copy()
        
        T *= cooling_rate
        if T < T_min:
            break

    # Reconstruct best solution from SA
    best_inner_sa = []
    for i in range(N):
        best_inner_sa.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
    best_ocx = best_params[N*3]
    best_ocy = best_params[N*3+1]
    best_oangle = best_params[N*3+2]
    
    # Verify disjointness
    if check_all_disjoint(best_inner_sa, eps=1e-5):
        best_inner = best_inner_sa
        best_outer_center = (best_ocx, best_ocy)
        best_outer_angle = best_oangle
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback to initial layout if SA broke it
        best_inner = initial_inner
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # 3. Polish Outer Container
    def objective_outer_polish(params):
        cx, cy, angle_deg = params
        return compute_required_side(best_inner, (cx, cy), angle_deg)
    
    best_polish_val = best_side
    best_polish_params = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
    
    # Try multiple starts
    for da in [-5, -2, 0, 2, 5]:
        if time.time() > deadline - 5:
            break
        start_p = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle + da])
        try:
            res = minimize(objective_outer_polish, start_p, method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 10000})
            if res.fun < best_polish_val:
                best_polish_val = res.fun
                best_polish_params = res.x
        except Exception:
            pass
            
    if best_polish_val < best_side:
        best_outer_center = (best_polish_params[0], best_polish_params[1])
        best_outer_angle = best_polish_params[2]
        best_side = best_polish_val

    # 4. Final Joint Polish (Small steps)
    current_params = np.array([v for row in best_inner for v in row] + [best_outer_center[0], best_outer_center[1], best_outer_angle])
    current_val = objective_joint(current_params)
    best_params_final = current_params.copy()
    best_val_final = current_val
    
    step_size = 0.05
    for _ in range(2000):
        if time.time() > deadline - 2:
            break
        
        neighbor = current_params.copy()
        indices = np.random.choice(len(neighbor), 3, replace=False)
        for idx in indices:
            neighbor[idx] += np.random.normal(0, step_size)
            
        neighbor_val = objective_joint(neighbor)
        
        if neighbor_val < current_val:
            current_params = neighbor
            current_val = neighbor_val
            if current_val < best_val_final:
                best_params_final = current_params.copy()
                best_val_final = current_val
        else:
            # Simple rejection
            pass
        step_size *= 0.99

    # Reconstruct final
    best_inner_final = []
    for i in range(N):
        best_inner_final.append([best_params_final[i*3], best_params_final[i*3+1], best_params_final[i*3+2]])
    best_ocx_final = best_params_final[N*3]
    best_ocy_final = best_params_final[N*3+1]
    best_oangle_final = best_params_final[N*3+2]
    
    if check_all_disjoint(best_inner_final, eps=1e-5):
        best_inner = best_inner_final
        best_outer_center = (best_ocx_final, best_ocy_final)
        best_outer_angle = best_oangle_final
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)
        
        # One last outer polish
        def objective_outer_final(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner, (cx, cy), angle_deg)
            
        try:
            res = minimize(objective_outer_final, [best_outer_center[0], best_outer_center[1], best_outer_angle],
                          method='Nelder-Mead', options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 20000})
            if res.fun < best_side:
                best_outer_center = (res.x[0], res.x[1])
                best_outer_angle = res.x[2]
                best_side = res.fun
        except Exception:
            pass

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
