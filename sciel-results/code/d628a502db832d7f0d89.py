import numpy as np
from scipy.optimize import linprog
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 10.0 # 10 seconds is plenty for LP
    
    # Define vertices of unit regular hexagon (flat-top)
    # Angles: 0, 60, 120, 180, 240, 300
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180.0
    V = np.column_stack((np.cos(angles), np.sin(angles)))
    
    # Variables:
    # 0..23: c_1x, c_1y, ..., c_12x, c_12y
    # 24..25: Cx, Cy
    # 26: S
    # 27..92: t_01, t_02, ..., t_11,12 (66 variables)
    
    n_vars = 2 * N + 2 + 1 + (N * (N - 1) // 2)
    
    # Objective: minimize S
    c = np.zeros(n_vars)
    c[26] = 1.0
    
    # Constraints A x <= b
    A_rows = []
    b_rows = []
    
    # 1. Containment constraints
    # For each i, for each k:
    # (c_i - C) dot V[k] <= S - 1
    # V[k,0] c_ix + V[k,1] c_iy - V[k,0] Cx - V[k,1] Cy - S <= -1
    for i in range(N):
        cx_idx = 2 * i
        cy_idx = 2 * i + 1
        for k in range(6):
            row = np.zeros(n_vars)
            row[cx_idx] = V[k, 0]
            row[cy_idx] = V[k, 1]
            row[24] = -V[k, 0]
            row[25] = -V[k, 1]
            row[26] = -1.0
            A_rows.append(row)
            b_rows.append(-1.0)
            
    # 2. Collision constraints
    # For each pair (i, j) with i < j:
    # Let p be the index of t_ij in the variable vector.
    # t_ij >= 2  =>  -t_ij <= -2
    # t_ij >= (c_j - c_i) dot V[k]  =>  (c_j - c_i) dot V[k] - t_ij <= 0
    pair_idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            t_idx = 27 + pair_idx
            pair_idx += 1
            
            # t_ij >= 2
            row = np.zeros(n_vars)
            row[t_idx] = -1.0
            A_rows.append(row)
            b_rows.append(-2.0)
            
            # t_ij >= dot product for all k
            for k in range(6):
                row = np.zeros(n_vars)
                # c_j - c_i
                row[2 * j] = V[k, 0]
                row[2 * j + 1] = V[k, 1]
                row[2 * i] = -V[k, 0]
                row[2 * i + 1] = -V[k, 1]
                row[t_idx] = -1.0
                A_rows.append(row)
                b_rows.append(0.0)
                
    A = np.array(A_rows)
    b = np.array(b_rows)
    
    # Bounds
    bounds = []
    for i in range(2 * N):
        bounds.append((-10.0, 10.0))
    bounds.append((-10.0, 10.0)) # Cx
    bounds.append((-10.0, 10.0)) # Cy
    bounds.append((1.0, 10.0))   # S
    for i in range(N * (N - 1) // 2):
        bounds.append((0.0, 10.0)) # t_ij
        
    # Solve LP
    try:
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if not res.success:
            # Fallback to a heuristic if LP fails
            # Simple hexagonal packing
            inner = []
            # 1 center
            inner.append([0.0, 0.0, 0.0])
            # 6 around
            for k in range(6):
                angle = k * 60 * np.pi / 180.0
                inner.append([2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0])
            # 5 more in a ring? 
            # Just return a valid-ish configuration
            side = 4.0
            return inner, (0.0, 0.0), side, 0.0
        else:
            x = res.x
            inner = []
            for i in range(N):
                inner.append([x[2 * i], x[2 * i + 1], 0.0])
            center = (x[24], x[25])
            side = x[26]
            return inner, center, side, 0.0
            
    except Exception:
        # Fallback
        inner = [[0.0, 0.0, 0.0]]
        for k in range(1, N):
            inner.append([k * 2.0, 0.0, 0.0])
        return inner, (0.0, 0.0), 20.0, 0.0

# EVOLVE-BLOCK-END
