import numpy as np
import time
from scipy.optimize import minimize

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 120s limit, keep margin
    
    # Initial positions: Hexagonal cluster
    sqrt3 = np.sqrt(3)
    # 7 hexes
    init_pos = [
        (0, 0),
        (sqrt3, 0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (-sqrt3, 0),
        (-sqrt3/2, -1.5),
        (sqrt3/2, -1.5)
    ]
    # Add 4 more in ring 2
    # (2*sqrt3, 0)
    # (sqrt3, 3)
    # (-sqrt3, 3)
    # (0, 3) -- wait, (0,3) is dist 3 from center. (sqrt3, 3) is dist sqrt(3+9)=sqrt(12).
    # Let's just place them in a 3x3 grid pattern roughly
    init_pos.append((2*sqrt3, 0))
    init_pos.append((sqrt3, 3))
    init_pos.append((-sqrt3, 3))
    init_pos.append((0, -3)) # Or (0,3)?
    
    # Flatten
    x0 = np.array(init_pos).flatten()
    
    def get_hex_vertices(cx, cy):
        angles = np.deg2rad(np.arange(0, 360, 60))
        xs = cx + np.cos(angles)
        ys = cy + np.sin(angles)
        return np.column_stack((xs, ys))

    def calc_outer_side(points):
        sqrt3_val = np.sqrt(3)
        normals = np.array([
            [1, 1/sqrt3_val],
            [-1, 1/sqrt3_val],
            [0, 2/sqrt3_val],
            [-1, -1/sqrt3_val],
            [1, -1/sqrt3_val],
            [0, -2/sqrt3_val]
        ])
        projections = points @ normals.T
        max_projections = np.max(projections, axis=1)
        return np.max(max_projections)

    def objective(pos_flat):
        positions = pos_flat.reshape(N, 2)
        
        # Calculate L
        all_vertices = []
        for i in range(N):
            all_vertices.append(get_hex_vertices(positions[i, 0], positions[i, 1]))
        all_vertices = np.vstack(all_vertices)
        L = calc_outer_side(all_vertices)
        
        # Calculate Overlap Penalty
        min_dist = np.sqrt(3) + 0.05
        penalty = 0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    penalty += (min_dist - d)**2
                    
        return L + 1000 * penalty

    best_result = None
    best_L = float('inf')
    
    # Run optimization
    # Use Nelder-Mead as it handles non-smooth max functions well
    options = {'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options=options)
        if res.success:
            pos = res.x.reshape(N, 2)
            # Verify no overlap
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < np.sqrt(3) - 1e-5:
                        valid = False
            if valid:
                # Calculate exact L
                all_vertices = []
                for i in range(N):
                    all_vertices.append(get_hex_vertices(pos[i, 0], pos[i, 1]))
                all_vertices = np.vstack(all_vertices)
                L = calc_outer_side(all_vertices)
                if L < best_L:
                    best_L = L
                    best_result = (pos, L)
    except Exception as e:
        pass

    if best_result is None:
        # Fallback: return initial config if it's valid
        pos = np.array(init_pos).reshape(N, 2)
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < np.sqrt(3):
                    valid = False
        if valid:
             all_vertices = []
             for i in range(N):
                 all_vertices.append(get_hex_vertices(pos[i, 0], pos[i, 1]))
             all_vertices = np.vstack(all_vertices)
             L = calc_outer_side(all_vertices)
             best_result = (pos, L)
        else:
             # Spread them out more
             pos = np.array(init_pos).reshape(N, 2) * 1.1
             all_vertices = []
             for i in range(N):
                 all_vertices.append(get_hex_vertices(pos[i, 0], pos[i, 1]))
             all_vertices = np.vstack(all_vertices)
             L = calc_outer_side(all_vertices)
             best_result = (pos, L)
             
    pos, L = best_result
    
    # Format output
    inner = [[pos[i, 0], pos[i, 1], 0.0] for i in range(N)]
    return inner, (0.0, 0.0), L, 0.0

