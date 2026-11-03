import numpy as np
import time
import math
import random

N = 12
DEADLINE = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START
def get_hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    """
    Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation.
    """
    rot_rad = np.radians(rot_deg)
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + rot_rad
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def min_dist_convex_poly(v1, v2):
    """
    Compute the minimum distance between two convex polygons.
    Returns a negative value if they overlap (penetration depth), 
    or positive distance if separated.
    """
    # For speed, we can first check bounding boxes
    # But SAT is robust.
    axes = []
    for v in (v1, v2):
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i+1)%len(v)]
            e = p2 - p1
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                axes.append(n)
    
    min_penetration = float('inf')
    for n in axes:
        proj1 = v1 @ n
        min1, max1 = np.min(proj1), np.max(proj1)
        proj2 = v2 @ n
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            # Separated
            gap = min2 - max1 if min2 > max1 else min1 - max2
            return gap
            
        overlap = min(max1, max2) - max(min1, min2)
        min_penetration = min(min_penetration, overlap)
        
    return -min_penetration

def calculate_outer_hex_side(points, outer_rot_deg):
    """
    Calculate the minimum side length L of a regular hexagon centered at (0,0)
    with rotation outer_rot_deg that encloses all given points.
    """
    angles = np.radians(np.array([90, 150, 210, 270, 330, 30]) + outer_rot_deg)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    
    proj = points @ normals.T
    max_proj = np.max(np.abs(proj))
    
    apothem = max_proj
    return (2.0 / np.sqrt(3.0)) * apothem

def objective(params, time_limit):
    """
    params: array of length 2*N + 1
    [x0, y0, ..., x11, y11, outer_rot_deg]
    """
    if time.time() > time_limit:
        return 1e10
    
    xys = params[:2*N].reshape(N, 2)
    outer_rot = params[2*N]
    
    # Compute vertices for all inner hexagons
    # Fix inner rotation to 0 for simplicity and speed
    vertices_list = [get_hex_vertices(xys[i][0], xys[i][1], 1.0, 0.0) for i in range(N)]
    
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dist = min_dist_convex_poly(vertices_list[i], vertices_list[j])
            if dist < 0:
                penalty += (-dist) ** 2
    
    all_points = np.vstack(vertices_list)
    L = calculate_outer_hex_side(all_points, outer_rot)
    
    return L + 1000.0 * penalty

def simulate_annealing(init_params, time_limit, initial_temp=1.0, cooling_rate=0.9995, steps=10000):
    """
    Run Simulated Annealing to minimize the objective function.
    """
    current_params = init_params.copy()
    current_score = objective(current_params, time_limit)
    
    best_params = current_params.copy()
    best_score = current_score
    
    temp = initial_temp
    n_vars = len(current_params)
    
    # Pre-allocate for speed
    for step in range(steps):
        if time.time() > time_limit - 5: # Stop a bit early to allow return
            break
            
        # Generate new solution
        # Randomly pick a few hexagons to move
        n_moves = random.randint(1, 3)
        new_params = current_params.copy()
        
        for _ in range(n_moves):
            idx = random.randint(0, N-1)
            # Move position
            # Step size proportional to temp
            dx = random.gauss(0, 0.5 * temp)
            dy = random.gauss(0, 0.5 * temp)
            new_params[2*idx] += dx
            new_params[2*idx+1] += dy
            
            # Occasionally adjust outer rotation
            if random.random() < 0.05:
                new_params[2*N] += random.gauss(0, 0.1 * temp)
        
        # Center the configuration to prevent drift
        # Calculate centroid of inner hexagons
        xys_new = new_params[:2*N].reshape(N, 2)
        centroid = np.mean(xys_new, axis=0)
        new_params[:2*N] -= centroid.flatten()
        
        new_score = objective(new_params, time_limit)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / (temp + 1e-9)):
            current_params = new_params
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_params = current_params.copy()
        
        temp *= cooling_rate
        if temp < 1e-6:
            temp = 1e-6
            
    return best_params, best_score

def construct():
    start_time = time.time()
    time_limit = start_time + DEADLINE - 5 # 140s
    
    best_params = None
    best_score = float('inf')
    
    # Initial configurations
    def make_brick_3x4():
        pts = []
        sqrt3 = np.sqrt(3)
        for r in range(4):
            y = r * 1.5
            for c in range(3):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2.0
                pts.append([x, y])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        return pts

    def make_brick_2x6():
        pts = []
        sqrt3 = np.sqrt(3)
        for r in range(2):
            y = r * 1.5
            for c in range(6):
                x = c * sqrt3
                if r % 2 == 1:
                    x += sqrt3 / 2.0
                pts.append([x, y])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        return pts

    def make_cluster():
        pts = []
        sqrt3 = np.sqrt(3)
        pts.append([0,0])
        for k in range(6):
            ang = np.radians(60*k)
            pts.append([sqrt3*np.cos(ang), sqrt3*np.sin(ang)])
        for k in range(5):
            ang = np.radians(60*k + 30)
            pts.append([2*sqrt3*np.cos(ang), 2*sqrt3*np.sin(ang)])
        pts = np.array(pts)
        pts -= np.mean(pts, axis=0)
        return pts

    configs = [
        make_brick_3x4(),
        make_brick_2x6(),
        make_cluster()
    ]
    
    # Run SA on multiple initial configs
    for cfg in configs:
        if time.time() > time_limit - 10:
            break
            
        init_params = np.concatenate((cfg.flatten(), [0.0]))
        
        # Run SA
        res_params, res_score = simulate_annealing(init_params, time_limit, initial_temp=0.5, cooling_rate=0.999, steps=5000)
        
        # Verify result
        xys = res_params[:2*N].reshape(N, 2)
        outer_rot = res_params[2*N]
        verts = [get_hex_vertices(xys[i][0], xys[i][1], 1.0, 0.0) for i in range(N)]
        
        # Check for overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if min_dist_convex_poly(verts[i], verts[j]) < -1e-4:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            L_val = calculate_outer_hex_side(np.vstack(verts), outer_rot)
            if L_val < best_score:
                best_score = L_val
                best_params = res_params.copy()
        
        # If we have a very good result, we can stop early or refine
        # For now, just run all configs within time limit

    # If no valid result found, return seed
    if best_params is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    xys = best_params[:2*N].reshape(N, 2)
    outer_rot = best_params[2*N]
    
    inner = []
    for i in range(N):
        inner.append([xys[i][0], xys[i][1], 0.0])
        
    return inner, (0.0, 0.0), best_score, outer_rot
# EVOLVE-BLOCK-END
