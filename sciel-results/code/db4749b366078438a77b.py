"""Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def polygons_overlap_sat(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Returns True if they overlap."""
    polys = [v1, v2]
    for poly in polys:
        for i in range(len(poly)):
            j = (i + 1) % len(poly)
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def compute_min_outer_side(inner_hexes, outer_center, outer_angle_deg):
    """
    Given inner hexagons (list of [cx, cy, angle_deg]) and outer hexagon params,
    find the minimum outer side length such that all inner hexagons fit inside.
    """
    ox, oy = outer_center
    oa = math.radians(outer_angle_deg)
    
    max_scale = 0.0
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = hex_vertices(cx, cy, ang)
        for vx, vy in verts:
            dx, dy = vx - ox, vy - oy
            required_s = 0.0
            for k in range(6):
                edge_mid_angle = oa + k * math.pi / 3 + math.pi / 6
                inward_nx = -math.cos(edge_mid_angle)
                inward_ny = -math.sin(edge_mid_angle)
                dot_val = dx * inward_nx + dy * inward_ny
                if dot_val > 0:
                    s_req = dot_val / (math.sqrt(3) / 2)
                    required_s = max(required_s, s_req)
            max_scale = max(max_scale, required_s)
    
    return max_scale

def objective(params, outer_angle_deg=0.0):
    """
    Objective function: sum of overlap penalties.
    params: array of length 3*N = [x0, y0, a0, x1, y1, a1, ...]
    Returns a scalar penalty value. Lower is better.
    """
    penalty = 0.0
    positions = params[0::3]
    ys = params[1::3]
    angles = params[2::3]
    
    # Precompute vertices for all hexagons
    verts_list = []
    for i in range(N):
        verts_list.append(hex_vertices(positions[i], ys[i], angles[i]))
    
    # 1. Inter-hexagon overlap penalty
    for i in range(N):
        for j in range(i + 1, N):
            v1 = verts_list[i]
            v2 = verts_list[j]
            if polygons_overlap_sat(v1, v2, tol=1e-8):
                # Calculate penetration depth approximation
                # Use center distance for a simple penalty
                dx = positions[i] - positions[j]
                dy = ys[i] - ys[j]
                dist = math.sqrt(dx*dx + dy*dy)
                # If overlapping, dist is likely small. Penalize heavily.
                # A better penalty: min distance between polygons.
                # For speed, we use a heuristic based on center distance.
                # Theoretical min distance between centers for non-overlapping unit hexagons is ~1.732 (sqrt(3)) if aligned.
                # Actually, for flat-top hexagons, min center distance is sqrt(3) * 1.0 = 1.732.
                # For pointy-top, it is also sqrt(3).
                # Let's use a soft penalty: max(0, (2.0 - dist)) ** 2
                # If dist < 2.0, there's a risk. If overlapping, dist is definitely < 2.0.
                # Let's use a stronger penalty if actually overlapping.
                penalty += 10.0 * max(0.0, 2.0 - dist) ** 2
    
    # 2. Container constraint penalty
    # We want to minimize the outer side length.
    # We can define the objective as the maximum required side length, plus overlap penalties.
    # However, 'minimize' in scipy works best with smooth functions.
    # Let's compute the required side length for the current configuration.
    inner_hexes = [[positions[i], ys[i], angles[i]] for i in range(N)]
    req_side = compute_min_outer_side(inner_hexes, (0.0, 0.0), outer_angle_deg)
    
    # We want to minimize req_side, but also ensure no overlaps.
    # So objective = req_side + penalty * large_weight
    # If there are overlaps, penalty is large, so we push to remove them.
    # Once no overlaps, penalty is 0, and we minimize req_side.
    
    total_penalty = req_side + penalty * 5.0
    
    return total_penalty

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # Safety margin (150s limit)

    sqrt3 = math.sqrt(3)
    
    # Initial setup: Grid
    positions = np.zeros(N)
    ys = np.zeros(N)
    angles = np.zeros(N)
    
    idx = 0
    for row in range(4):
        for col in range(3):
            cx = col * 1.5
            cy = row * sqrt3
            if row % 2 == 1:
                cx += 0.75
            positions[idx] = cx - 1.5
            ys[idx] = cy - 1.5 * sqrt3
            angles[idx] = 0.0
            idx += 1
            
    # Combine into a single parameter vector
    x0 = np.zeros(3 * N)
    for i in range(N):
        x0[3*i] = positions[i]
        x0[3*i+1] = ys[i]
        x0[3*i+2] = angles[i]

    best_x = x0.copy()
    best_score = float('inf')
    
    # Multi-start local optimization using Nelder-Mead
    # Nelder-Mead is derivative-free and robust for non-smooth objectives.
    
    num_starts = 20
    rng = np.random.default_rng(42)
    
    for start_idx in range(num_starts):
        if time.time() > deadline:
            break
            
        # Perturb initial solution
        if start_idx == 0:
            current_x = x0.copy()
        else:
            current_x = x0 + rng.normal(0, 1.0, size=3*N)
            
        # Define bounds? Nelder-Mead doesn't support bounds directly.
        # We rely on the penalty function to keep things reasonable.
        
        # Optimize
        try:
            res = minimize(objective, current_x, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.success or res.fun < best_score:
                # Check validity: ensure no overlaps in the final result
                pos = res.x[0::3]
                yy = res.x[1::3]
                ang = res.x[2::3]
                
                verts_list = [hex_vertices(pos[i], yy[i], ang[i]) for i in range(N)]
                
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if polygons_overlap_sat(verts_list[i], verts_list[j], tol=1e-6):
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    score = res.fun
                    # Recompute exact side length to be sure
                    inner_hexes = [[pos[i], yy[i], ang[i]] for i in range(N)]
                    exact_side = compute_min_outer_side(inner_hexes, (0.0, 0.0), 0.0)
                    # The objective function included penalty, so if penalty is 0, score == exact_side
                    # But let's use exact_side for the record
                    if exact_side < best_score:
                        best_score = exact_side
                        best_x = res.x.copy()
                        
        except Exception as e:
            pass # Continue to next start

    # If no valid solution was found (shouldn't happen with grid init), fallback
    if best_score == float('inf'):
        pos = x0[0::3]
        yy = x0[1::3]
        ang = x0[2::3]
        inner_hexes = [[pos[i], yy[i], ang[i]] for i in range(N)]
        best_score = compute_min_outer_side(inner_hexes, (0.0, 0.0), 0.0)
        best_x = x0.copy()

    # Extract final solution
    pos = best_x[0::3]
    yy = best_x[1::3]
    ang = best_x[2::3]
    
    inner = []
    for i in range(N):
        inner.append([pos[i], yy[i], math.degrees(ang[i])])
    
    return inner, (0.0, 0.0), best_score, 0.0

# EVOLVE-BLOCK-END
