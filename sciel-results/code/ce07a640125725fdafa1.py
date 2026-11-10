import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0

    # Unit hexagon vertices (angle 0)
    # Vertices at 0, 60, ... 300 degrees
    hex_verts = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(0, 360, 60)
    ])

    # Normals for unit hexagon (angle 0)
    # Normals point to midpoints of sides, angles 30, 90, ...
    hex_normals = np.array([
        [np.cos(np.deg2rad(a)), np.sin(np.deg2rad(a))] for a in range(30, 360, 60)
    ])

    # Function to calculate the side length of the container
    def calc_side(centers, theta):
        # Rotate normals by theta
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        normals = (R @ hex_normals.T).T
        
        # Project all inner hexagon vertices onto these normals
        # centers: (11, 2)
        # hex_verts: (6, 2)
        # We need max over i, j of (C_i + V_j) . n_k
        # This is equivalent to max_j (V_j . n_k) + max_i (C_i . n_k) ??
        # No, max over i and j is NOT sum of maxes.
        # But since the set of vertices is the same for all hexagons,
        # max_{i,j} (C_i + V_j) . n_k = max_i (C_i . n_k) + max_j (V_j . n_k)
        # Wait. (A+B).n = A.n + B.n.
        # max_{i,j} (x_i + y_j) = max_i x_i + max_j y_j.
        # Yes!
        
        proj_centers = centers @ normals.T  # (11, 6)
        max_proj_centers = np.max(proj_centers, axis=0) # (6,)
        
        proj_hex = hex_verts @ normals.T # (6, 6)
        max_proj_hex = np.max(proj_hex, axis=0) # (6,)
        
        max_extents = max_proj_centers + max_proj_hex # (6,)
        max_extent = np.max(max_extents)
        
        side = (2.0 / np.sqrt(3.0)) * max_extent
        return side

    # Function to check overlaps
    # Returns penalty if any pair overlaps
    def overlap_penalty(centers):
        penalty = 0.0
        n = len(centers)
        # Precompute center differences
        # For aligned hexagons, overlap if max_k |(C_i - C_j) . n_k| <= sqrt(3)
        # Actually, the exclusion zone apothem is sqrt(3).
        # So if max_k |proj| < sqrt(3) - epsilon, they overlap.
        # We use a smooth penalty.
        
        for i in range(n):
            for j in range(i+1, n):
                d = centers[i] - centers[j]
                projs = np.abs(d @ hex_normals.T)
                max_proj = np.max(projs)
                # If max_proj < sqrt(3), they are inside the exclusion zone
                # The "depth" of overlap is roughly sqrt(3) - max_proj
                if max_proj < np.sqrt(3):
                    # Penalty proportional to how deep they are
                    # Use a quadratic penalty for smoothness
                    penalty += (np.sqrt(3) - max_proj) ** 2
        return penalty

    # Objective function
    def objective(vars):
        # vars: x1, y1, ..., x4, y4, theta
        # 4 outer hexagons + 1 angle
        if time.time() > deadline - 5:
            return float('inf')
            
        x1, y1, x2, y2, x3, y3, x4, y4, theta = vars
        
        # Fixed centers
        d = np.sqrt(3)
        c0 = [0, 0]
        c1 = [d * np.cos(0), d * np.sin(0)]
        c2 = [d * np.cos(np.pi/3), d * np.sin(np.pi/3)]
        c3 = [d * np.cos(2*np.pi/3), d * np.sin(2*np.pi/3)]
        c4 = [d * np.cos(np.pi), d * np.sin(np.pi)]
        c5 = [d * np.cos(4*np.pi/3), d * np.sin(4*np.pi/3)]
        c6 = [d * np.cos(5*np.pi/3), d * np.sin(5*np.pi/3)]
        
        c7 = [x1, y1]
        c8 = [x2, y2]
        c9 = [x3, y3]
        c10 = [x4, y4]
        
        centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
        
        side = calc_side(centers, theta)
        pen = overlap_penalty(centers)
        
        return side + 1000.0 * pen

    # Initial guesses for the 4 outer hexagons
    # We try a few symmetric patterns
    # Pattern 1: Cross (0, 90, 180, 270) at distance 2*sqrt(3)
    d2 = 2 * np.sqrt(3)
    guesses = []
    
    # Guess 1: 0, 90, 180, 270
    guesses.append([
        d2 * np.cos(0), d2 * np.sin(0),
        d2 * np.cos(np.pi/2), d2 * np.sin(np.pi/2),
        d2 * np.cos(np.pi), d2 * np.sin(np.pi),
        d2 * np.cos(3*np.pi/2), d2 * np.sin(3*np.pi/2),
        0.0
    ])
    
    # Guess 2: 30, 120, 210, 300
    guesses.append([
        d2 * np.cos(np.pi/6), d2 * np.sin(np.pi/6),
        d2 * np.cos(2*np.pi/3), d2 * np.sin(2*np.pi/3),
        d2 * np.cos(7*np.pi/6), d2 * np.sin(7*np.pi/6),
        d2 * np.cos(5*np.pi/3), d2 * np.sin(5*np.pi/3),
        0.0
    ])
    
    # Guess 3: 60, 150, 240, 330
    guesses.append([
        d2 * np.cos(np.pi/3), d2 * np.sin(np.pi/3),
        d2 * np.cos(5*np.pi/6), d2 * np.sin(5*np.pi/6),
        d2 * np.cos(4*np.pi/3), d2 * np.sin(4*np.pi/3),
        d2 * np.cos(11*np.pi/6), d2 * np.sin(11*np.pi/6),
        0.0
    ])
    
    # Guess 4: 0, 60, 180, 240
    guesses.append([
        d2 * np.cos(0), d2 * np.sin(0),
        d2 * np.cos(np.pi/3), d2 * np.sin(np.pi/3),
        d2 * np.cos(np.pi), d2 * np.sin(np.pi),
        d2 * np.cos(4*np.pi/3), d2 * np.sin(4*np.pi/3),
        0.0
    ])
    
    best_side = float('inf')
    best_config = None
    
    for guess in guesses:
        if time.time() > deadline - 30:
            break
            
        try:
            res = minimize(objective, guess, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_side and res.fun < 100: # Sanity check
                best_side = res.fun
                # Extract config
                x1, y1, x2, y2, x3, y3, x4, y4, theta = res.x
                
                d = np.sqrt(3)
                c0 = [0, 0]
                c1 = [d * np.cos(0), d * np.sin(0)]
                c2 = [d * np.cos(np.pi/3), d * np.sin(np.pi/3)]
                c3 = [d * np.cos(2*np.pi/3), d * np.sin(2*np.pi/3)]
                c4 = [d * np.cos(np.pi), d * np.sin(np.pi)]
                c5 = [d * np.cos(4*np.pi/3), d * np.sin(4*np.pi/3)]
                c6 = [d * np.cos(5*np.pi/3), d * np.sin(5*np.pi/3)]
                
                c7 = [x1, y1]
                c8 = [x2, y2]
                c9 = [x3, y3]
                c10 = [x4, y4]
                
                centers = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10])
                
                # Verify no overlap
                if overlap_penalty(centers) < 1e-6:
                    # Calculate exact side
                    side = calc_side(centers, theta)
                    if side < best_side:
                        best_side = side
                        inner_hex_data = np.column_stack([centers, np.zeros(11)])
                        outer_hex_data = np.array([0, 0, np.rad2deg(theta)])
                        best_config = (inner_hex_data, outer_hex_data, side)
        except Exception as e:
            pass

    if best_config is None:
        # Fallback to a known valid configuration
        # Simple grid
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

    return best_config[0], best_config[1], best_config[2]
