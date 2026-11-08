"""Packing 12 unit hexagons in a regular hexagon; minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy)."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def min_outer_side_for_points(points, outer_center, outer_angle_deg):
    """
    Compute minimum side length s of a regular hexagon centered at outer_center
    with rotation outer_angle_deg that contains all given points.
    """
    cx, cy = outer_center
    angle_rad = np.radians(outer_angle_deg)
    normal_angles = angle_rad + np.pi/6 + np.arange(6) * np.pi/3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    rel_points = points - np.array([cx, cy])
    dots = rel_points @ normals.T
    max_dot = np.max(dots, axis=0)
    r_needed = np.max(max_dot)
    if r_needed < 0:
        r_needed = 0.0
    s = 2.0 / np.sqrt(3.0) * r_needed
    return s

def compute_min_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length for given inner configuration."""
    all_points = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        all_points.append(v)
    all_points = np.vstack(all_points)
    return min_outer_side_for_points(all_points, outer_center, outer_angle_deg)

def _polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex hexagons (given as (6,2) arrays) have overlapping interiors."""
    for verts in [v1, v2]:
        for k in range(6):
            edge = verts[(k+1)%6] - verts[k]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal = normal / np.linalg.norm(normal)
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 - tol or max2 < min1 - tol:
                return False
    return True

def check_disjoint(inner, tol=1e-6):
    """Check if all inner hexagons are disjoint (no overlapping interiors)."""
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            if _polygons_overlap(v1, v2, tol=tol):
                return False
    return True

def sat_overlap_penalty(v1, v2, strength=1000.0):
    """Compute overlap penalty using SAT. Returns 0 if not overlapping."""
    min_overlap = 0.0
    for verts in [v1, v2]:
        for k in range(6):
            edge = verts[(k+1)%6] - verts[k]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            proj1 = np.dot(v1, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > min_overlap:
                min_overlap = overlap
    if min_overlap <= 0:
        return 0.0
    return strength * min_overlap * min_overlap

def objective(params):
    """
    Objective: minimize outer side length + penalty for overlaps.
    params: 36 values [x1,y1,a1, x2,y2,a2, ..., x12,y12,a12]
    """
    pos = params[:24].reshape(N, 2)
    angles = params[24:36]
    
    all_points = []
    for i in range(N):
        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
        all_points.append(v)
    all_points = np.vstack(all_points)
    
    s = min_outer_side_for_points(all_points, (0.0, 0.0), 0.0)
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            v1 = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
            v2 = hexagon_vertices(pos[j, 0], pos[j, 1], 1.0, angles[j])
            penalty += sat_overlap_penalty(v1, v2, strength=500.0)
    
    return s + penalty

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0
    
    def try_config(inner, center, angle):
        nonlocal best_score, best_inner, best_center, best_side, best_angle
        s = compute_min_side(inner, center, angle)
        if s < best_score:
            best_score = s
            best_inner = [list(x) for x in inner]
            best_center = center
            best_side = s
            best_angle = angle
    
    sqrt3 = np.sqrt(3)
    
    # Initial heuristic configurations
    configs = []
    
    # Config 1: 3x4 hexagonal grid
    positions = []
    for i in range(3):
        for j in range(4):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions.append((x, y))
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    configs.append([[p[0] - cx, p[1] - cy, 0.0] for p in positions])
    
    # Config 2: 4x3 hexagonal grid
    positions2 = []
    for i in range(4):
        for j in range(3):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions2.append((x, y))
    cx2 = np.mean([p[0] for p in positions2])
    cy2 = np.mean([p[1] for p in positions2])
    configs.append([[p[0] - cx2, p[1] - cy2, 0.0] for p in positions2])
    
    # Config 3: 2x6 grid
    positions3 = []
    for i in range(2):
        for j in range(6):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions3.append((x, y))
    cx3 = np.mean([p[0] for p in positions3])
    cy3 = np.mean([p[1] for p in positions3])
    configs.append([[p[0] - cx3, p[1] - cy3, 0.0] for p in positions3])
    
    # Config 4: Center + ring
    positions4 = [(0, 0)]
    for k in range(6):
        ang = k * np.pi / 3
        positions4.append((2*sqrt3*np.cos(ang), 2*sqrt3*np.sin(ang)))
    for k in range(5):
        ang = k * 2*np.pi/5 + np.pi/10
        positions4.append((4*sqrt3*np.cos(ang), 4*sqrt3*np.sin(ang)))
    cx4 = np.mean([p[0] for p in positions4])
    cy4 = np.mean([p[1] for p in positions4])
    configs.append([[p[0] - cx4, p[1] - cy4, 0.0] for p in positions4])
    
    # Try all configs with various outer rotations
    for cfg in configs:
        if time.time() - start_time > DEADLINE:
            break
        for outer_ang in [0, 30, 60, 90, 120, 150]:
            if time.time() - start_time > DEADLINE:
                break
            try_config(cfg, (0.0, 0.0), float(outer_ang))
    
    # Physics-based optimization with multiple restarts
    if best_inner is not None and time.time() - start_time < DEADLINE - 10:
        # Use the best initial config as starting point
        init_params = np.zeros(36)
        for i in range(N):
            init_params[i*2] = best_inner[i][0]
            init_params[i*2+1] = best_inner[i][1]
            init_params[i*2+2] = best_inner[i][2]
        
        n_restarts = 8
        for r in range(n_restarts):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            if r == 0:
                params = init_params.copy()
            else:
                params = init_params.copy()
                noise_scale = 0.1 * (r + 1)
                params[:24] += noise_scale * np.random.randn(24)
                params[24:36] += noise_scale * np.random.randn(12) * 10  # angle noise in degrees
            
            # L-BFGS-B optimization
            try:
                result = minimize(
                    objective,
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                pos_opt = result.x[:24].reshape(N, 2)
                ang_opt = result.x[24:36]
                
                inner_opt = [[float(pos_opt[i, 0]), float(pos_opt[i, 1]), float(ang_opt[i])] for i in range(N)]
                
                # Check disjointness
                if check_disjoint(inner_opt, tol=1e-5):
                    s_opt = compute_min_side(inner_opt, (0.0, 0.0), 0.0)
                    if s_opt < best_score:
                        best_score = s_opt
                        best_inner = inner_opt
                        best_center = (0.0, 0.0)
                        best_side = s_opt
                        best_angle = 0.0
            except Exception:
                pass
    
    # Final polish: try to tighten with smaller steps
    if best_inner is not None and time.time() - start_time < DEADLINE - 5:
        init_params = np.zeros(36)
        for i in range(N):
            init_params[i*2] = best_inner[i][0]
            init_params[i*2+1] = best_inner[i][1]
            init_params[i*2+2] = best_inner[i][2]
        
        try:
            result = minimize(
                objective,
                init_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            pos_opt = result.x[:24].reshape(N, 2)
            ang_opt = result.x[24:36]
            
            inner_opt = [[float(pos_opt[i, 0]), float(pos_opt[i, 1]), float(ang_opt[i])] for i in range(N)]
            
            if check_disjoint(inner_opt, tol=1e-5):
                s_opt = compute_min_side(inner_opt, (0.0, 0.0), 0.0)
                if s_opt < best_score:
                    best_score = s_opt
                    best_inner = inner_opt
                    best_center = (0.0, 0.0)
                    best_side = s_opt
                    best_angle = 0.0
        except Exception:
            pass
    
    # Add small safety margin
    best_side += 0.02
    
    return best_inner, best_center, best_side, best_angle
