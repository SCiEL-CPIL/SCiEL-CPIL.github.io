import numpy as np
from scipy.optimize import minimize_scalar
from itertools import combinations
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin

    # Hexagonal lattice points for k=3 cluster (19 points)
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                points.append((q, r))
    
    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    cart_points = [axial_to_cartesian(q, r) for q, r in points]
    
    core_indices = []
    outer_indices = []
    for i, (q, r) in enumerate(points):
        dist = (abs(q) + abs(r) + abs(q + r)) / 2
        if dist <= 1:
            core_indices.append(i)
        else:
            outer_indices.append(i)
            
    best_side = float('inf')
    best_config = None
    best_alpha = 0.0

    angles = np.deg2rad(np.arange(0, 360, 60))
    base_vertices = np.column_stack((np.cos(angles), np.sin(angles)))
    
    count = 0
    for combo in combinations(outer_indices, 5):
        if time.time() > deadline:
            break
        
        selected_indices = core_indices + list(combo)
        
        all_vertices = np.zeros((72, 2))
        for i, idx in enumerate(selected_indices):
            cx, cy = cart_points[idx]
            verts = base_vertices + np.array([cx, cy])
            all_vertices[i*6:(i+1)*6] = verts
        
        def obj(alpha):
            normal_angles = alpha + np.deg2rad(30 + np.arange(0, 360, 60))
            normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
            projs = all_vertices @ normals.T
            max_projs = np.max(projs, axis=0)
            return np.max(max_projs)

        res = minimize_scalar(obj, bounds=(0, np.pi/3), method='bounded')
        
        if res.success:
            max_proj_val = res.fun
            side = (2.0 / np.sqrt(3)) * max_proj_val
            
            if side < best_side:
                best_side = side
                best_config = selected_indices
                best_alpha = res.x
        count += 1

    if best_config is None:
        # Fallback to seed-like behavior if optimization fails or times out
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for idx in best_config:
        x, y = cart_points[idx]
        inner.append([x, y, 0.0])
    
    return inner, (0.0, 0.0), best_side, np.rad2deg(best_alpha)
# EVOLVE-BLOCK-END
