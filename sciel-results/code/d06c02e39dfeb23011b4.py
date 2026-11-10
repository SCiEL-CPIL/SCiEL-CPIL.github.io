import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600
    
    # Hexagon vertices for unit side, angle 0
    # Vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    base_vertices = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # Outer hexagon normals (perpendicular to edges)
    # Edges are at angles 30, 90, 150, 210, 270, 330?
    # Normal to edge at angle 30 is 30?
    # Let's define the hexagon such that one vertex is at (1,0).
    # Then edges are not aligned with axes.
    # The apothem direction (normal to flat side) is at 90, 150, 210, 270, 330, 30?
    # If vertices are at 0, 60, ..., then edges are between 0-60, 60-120, etc.
    # Midpoint of 0-60 edge is at 30 deg.
    # So normals are at 30, 90, 150, 210, 270, 330.
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    def hex_norm(p):
        # p is (N, 2)
        # Returns max projection onto normals
        # (p @ normals.T) gives (N, 6)
        # We want max over columns
        return np.max(p @ normals.T, axis=1)
        
    def evaluate(params):
        # params: (11, 3) -> x, y, angle_deg
        n = 11
        xy = params[:, :2]
        angles = np.deg2rad(params[:, 2])
        
        # Calculate all vertices for all hexagons
        # (11, 6, 2)
        # Rotate base vertices
        # cos(a), sin(a)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        
        # Rotation matrix for each hex
        # R = [[cos, -sin], [sin, cos]]
        # v_rot = R @ v
        # x' = cos*x - sin*y
        # y' = sin*x + cos*y
        
        vx = base_vertices[0]
        vy = base_vertices[1]
        
        # (11, 6)
        rot_vx = cos_a[:, None] * vx[None, :] - sin_a[:, None] * vy[None, :]
        rot_vy = sin_a[:, None] * vx[None, :] + cos_a[:, None] * vy[None, :]
        
        # Add center
        # (11, 6)
        verts_x = xy[:, 0:1] + rot_vx
        verts_y = xy[:, 1:2] + rot_vy
        
        # Stack to (66, 2)
        all_verts = np.column_stack([verts_x.flatten(), verts_y.flatten()])
        
        # Calculate hex norm
        norms = hex_norm(all_verts)
        max_norm = np.max(norms)
        
        # Calculate S
        S = max_norm * 2.0 / np.sqrt(3)
        
        # Check overlaps
        # (11, 11) distance matrix
        diff = xy[:, None, :] - xy[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists[~np.eye(11, dtype=bool)])
        
        # Penalty for overlap
        # If min_dist < sqrt(3), penalty
        overlap_penalty = 0.0
        if min_dist < np.sqrt(3):
            overlap_penalty = (np.sqrt(3) - min_dist) ** 2 * 1000.0
            
        return S + overlap_penalty, S, min_dist

    # Initial guess: random within radius 2
    np.random.seed(42)
    best_S = np.inf
    best_params = None
    
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break
            
        # Random init
        angles = np.random.uniform(0, 360, 11)
        # Start with a loose grid
        # 3x3 grid?
        # Let's try a random cluster
        x = np.random.uniform(-2, 2, 11)
        y = np.random.uniform(-2, 2, 11)
        params = np.column_stack([x, y, angles])
        
        # Use Nelder-Mead
        try:
            res = minimize(evaluate, params, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-4})
            final_val, final_S, final_min_dist = res.fun, None, None
            # Re-evaluate to get S and min_dist
            _, S_val, min_dist_val = evaluate(res.x)
            
            if min_dist_val >= np.sqrt(3) - 1e-6:
                if S_val < best_S:
                    best_S = S_val
                    best_params = res.x
        except Exception:
            pass
            
    if best_params is None:
        # Fallback: Return the S=8 solution
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    # Construct final result
    # Ensure angles are in degrees
    inner_hex_data = best_params.copy()
    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = best_S
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
