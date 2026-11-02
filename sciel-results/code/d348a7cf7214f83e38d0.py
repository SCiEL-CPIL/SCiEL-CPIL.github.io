"""Packing 11 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    sqrt3 = np.sqrt(3.0)
    
    # Initial configurations: different cluster topologies
    modes = []
    
    # Mode 0: Hexagonal cluster - center + 6 ring + 4 outer
    pos0 = [
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, sqrt3],
        [-1.0, sqrt3],
        [-2.0, 0.0],
        [-1.0, -sqrt3],
        [1.0, -sqrt3],
        [3.0, sqrt3],
        [3.0, -sqrt3],
        [-3.0, sqrt3],
        [-3.0, -sqrt3],
    ]
    modes.append(np.array(pos0))
    
    # Mode 1: Cross-like arrangement
    pos1 = [
        [0.0, 0.0],
        [2.0, 0.0],
        [1.0, sqrt3],
        [-1.0, sqrt3],
        [-2.0, 0.0],
        [-1.0, -sqrt3],
        [1.0, -sqrt3],
        [4.0, 0.0],
        [0.0, 2.0*sqrt3],
        [0.0, -2.0*sqrt3],
        [-4.0, 0.0],
    ]
    modes.append(np.array(pos1))
    
    # Mode 2: Bent line
    pos2 = [
        [0.0, 0.0],
        [2.0, 0.0],
        [4.0, 0.0],
        [1.0, sqrt3],
        [3.0, sqrt3],
        [-1.0, sqrt3],
        [0.0, 2.0*sqrt3],
        [-2.0, 0.0],
        [-1.0, -sqrt3],
        [1.0, -sqrt3],
        [3.0, -sqrt3],
    ]
    modes.append(np.array(pos2))
    
    best_result = None
    best_score = np.inf
    
    for mode_idx, init_pos in enumerate(modes):
        if time.time() > deadline:
            break
        
        for theta_init in [0.0, np.pi/6, np.pi/3]:
            if time.time() > deadline:
                break
            
            params = np.zeros(34)
            for i in range(N):
                params[i*3] = init_pos[i][0]
                params[i*3+1] = init_pos[i][1]
                params[i*3+2] = 0.0
            params[33] = theta_init
            
            def obj(params_radians):
                positions = np.zeros((N, 2))
                angles_rad = np.zeros(N)
                for i in range(N):
                    positions[i] = params_radians[i*3:i*3+2]
                    angles_rad[i] = params_radians[i*3+2]
                theta_outer = params_radians[33]
                
                vertices = []
                for i in range(N):
                    vx, vy = positions[i]
                    angles = angles_rad[i] + np.arange(6) * np.pi / 3.0
                    verts = np.column_stack([np.cos(angles), np.sin(angles)])
                    vertices.append(verts + np.array([vx, vy]))
                vertices = np.vstack(vertices)
                
                normals = []
                for k in range(6):
                    ang = theta_outer + k * np.pi / 3.0
                    normals.append(np.array([np.cos(ang), np.sin(ang)]))
                normals = np.array(normals)
                
                projections = vertices @ normals.T
                max_proj = np.max(projections, axis=0)
                max_abs_proj = np.max(np.abs(max_proj))
                s = 2.0 * max_abs_proj / np.sqrt(3.0)
                return s
            
            try:
                result = minimize(
                    obj,
                    params,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.fun < best_score:
                    final_params = result.x
                    final_score = result.fun
                    
                    inner = []
                    for i in range(N):
                        x = final_params[i*3]
                        y = final_params[i*3+1]
                        angle_deg = np.degrees(final_params[i*3+2])
                        inner.append([x, y, angle_deg])
                    
                    outer_center = [0.0, 0.0]
                    outer_side = final_score
                    outer_angle_deg = np.degrees(final_params[33])
                    
                    best_score = final_score
                    best_result = (inner, outer_center, outer_side, outer_angle_deg)
                    
            except Exception:
                pass
    
    if best_result is None:
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, [0.0, 0.0], 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
