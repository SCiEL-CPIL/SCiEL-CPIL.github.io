import numpy as np
import time
import random
import scipy.linalg

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    # Known rank-42 decomposition (Strassen-like)
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # The 7 "steps" each expand into 6 columns (2 x-blocks × 3 y-blocks), total 42
    steps = [
        (
            {(0,0): 1, (0,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1), (3, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (1,0): 1, (1,1): 1, (1,2): 1},
            [(2, 1), (3, -1)]
        ),
        (
            {(0,0): 1, (0,1): 1},
            {(0,3): 1, (0,4): 1, (0,5): 1, (1,3): 1, (1,4): 1, (1,5): 1, (2,3): -1, (2,4): -1, (2,5): -1, (3,3): -1, (3,4): -1, (3,5): -1},
            [(1, 1), (3, 1)]
        ),
        (
            {(1,2): 1, (1,3): 1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (3,0): 1, (3,1): 1, (3,2): 1, (0,0): -1, (0,1): -1, (0,2): -1, (1,0): -1, (1,1): -1, (1,2): -1},
            [(0, 1), (2, 1)]
        ),
        (
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1},
            {(2,3): 1, (2,4): 1, (2,5): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, -1), (1, 1)]
        ),
        (
            {(1,0): 1, (1,1): 1, (0,0): -1, (0,1): -1},
            {(0,0): 1, (0,1): 1, (0,2): 1, (0,3): 1, (0,4): 1, (0,5): 1, (1,0): 1, (1,1): 1, (1,2): 1, (1,3): 1, (1,4): 1, (1,5): 1},
            [(3, 1)]
        ),
        (
            {(0,2): 1, (0,3): 1, (1,2): -1, (1,3): -1},
            {(2,0): 1, (2,1): 1, (2,2): 1, (2,3): 1, (2,4): 1, (2,5): 1, (3,0): 1, (3,1): 1, (3,2): 1, (3,3): 1, (3,4): 1, (3,5): 1},
            [(0, 1)]
        ),
    ]
    
    col = 0
    for X_dict, Y_dict, C_targets in steps:
        # Split X by parity of j (column index in the 4-wide middle dim)
        x0 = {k: v for k, v in X_dict.items() if k[1] % 2 == 0}
        x1 = {k: v for k, v in X_dict.items() if k[1] % 2 == 1}
        
        # Split Y by (j%2, k%3) into 6 buckets
        y_buckets = [[], [], [], [], [], []]
        for (j, k), c in Y_dict.items():
            idx = (j % 2) * 3 + (k % 3)
            y_buckets[idx].append(((j, k), c))
        
        # 6 sub-columns: (x0,y0,z0), (x1,y3,z0), (x0,y1,z1), (x1,y4,z1), (x0,y2,z2), (x1,y5,z2)
        sub_terms = [
            (x0, y_buckets[0], 0),
            (x1, y_buckets[3], 0),
            (x0, y_buckets[1], 1),
            (x1, y_buckets[4], 1),
            (x0, y_buckets[2], 2),
            (x1, y_buckets[5], 2),
        ]
        
        for X_c, Y_c, z_idx in sub_terms:
            for (i, j), c in X_c.items():
                U[i * m + j, col] += c
            for (j, k), c in Y_c:
                V[j * p + k, col] += c
            
            for block_idx, sign in C_targets:
                if block_idx == 0:
                    i_C, k_C = 0, z_idx
                elif block_idx == 1:
                    i_C, k_C = 0, 3 + z_idx
                elif block_idx == 2:
                    i_C, k_C = 1, z_idx
                elif block_idx == 3:
                    i_C, k_C = 1, 3 + z_idx
                W[k_C * n + i_C, col] += sign
            
            col += 1
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Now try to reduce rank by finding linear dependencies among columns
    # Use scipy.linalg.null_space for better robustness
    
    nU = n * m  # 8
    nV = m * p  # 24
    nW = p * n  # 12
    
    # Build the tensor evaluation matrix
    # Vectorized construction
    # U: (8, 42), V: (24, 42), W: (12, 42)
    # T_eval: (2304, 42)
    # Index mapping: row = i * 24 + j * 6 + k?
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # We can construct T_eval using broadcasting
    # But 2304 x 42 is small enough to loop if needed, but let's try to vectorize.
    
    # Actually, the current loop is fine.
    
    # Find null space
    tol = 1e-10
    null_dim = 0
    null_space = np.zeros((0, rank))
    
    try:
        rows = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i*m+j, :]
                    v_val = V[j*p+k, :]
                    w_val = W[k*n+i, :]
                    row = u_val * v_val * w_val
                    rows.append(row)
        
        T_eval = np.array(rows)
        
        # Use scipy.linalg.null_space
        null_space = scipy.linalg.null_space(T_eval, rcond=1e-10)
        null_dim = null_space.shape[1]
    except Exception:
        null_dim = 0
        null_space = np.zeros((0, rank))
    
    if null_dim > 0 and time.time() < deadline:
        # Eliminate dependencies
        for _ in range(null_dim):
            if time.time() > deadline:
                break
            
            # Find column with largest |alpha| in the first null vector
            # We should pick the null vector that allows for the best elimination
            # Simply using the first one is a heuristic
            
            best_null = null_space[:, 0]
            elim_idx = int(np.argmax(np.abs(best_null)))
            alpha_elim = best_null[elim_idx]
            
            if abs(alpha_elim) < 1e-15:
                # Try next null vector or break
                continue
            
            # Update U, V, W
            for l in range(rank):
                if l == elim_idx:
                    continue
                coeff = -best_null[l] / alpha_elim
                U[:, l] += coeff * U[:, elim_idx]
                V[:, l] += coeff * V[:, elim_idx]
                W[:, l] += coeff * W[:, elim_idx]
            
            # Remove column
            mask = [i for i in range(rank) if i != elim_idx]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            rank -= 1
            
            if rank <= 0:
                break
            
            # Recompute null space
            try:
                rows = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_val = U[i*m+j, :]
                            v_val = V[j*p+k, :]
                            w_val = W[k*n+i, :]
                            row = u_val * v_val * w_val
                            rows.append(row)
                
                T_eval = np.array(rows)
                null_space = scipy.linalg.null_space(T_eval, rcond=1e-10)
                null_dim = null_space.shape[1]
            except Exception:
                break
            
            if null_dim == 0:
                break
    
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] for l in range(rank))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-8:
        # Fallback
        U = np.zeros((n * m, 42), dtype=float)
        V = np.zeros((m * p, 42), dtype=float)
        W = np.zeros((p * n, 42), dtype=float)
        
        col = 0
        for X_dict, Y_dict, C_targets in steps:
            x0 = {k: v for k, v in X_dict.items() if k[1] % 2 == 0}
            x1 = {k: v for k, v in X_dict.items() if k[1] % 2 == 1}
            
            y_buckets = [[], [], [], [], [], []]
            for (j, k), c in Y_dict.items():
                idx = (j % 2) * 3 + (k % 3)
                y_buckets[idx].append(((j, k), c))
            
            sub_terms = [
                (x0, y_buckets[0], 0),
                (x1, y_buckets[3], 0),
                (x0, y_buckets[1], 1),
                (x1, y_buckets[4], 1),
                (x0, y_buckets[2], 2),
                (x1, y_buckets[5], 2),
            ]
            
            for X_c, Y_c, z_idx in sub_terms:
                for (i, j), c in X_c.items():
                    U[i * m + j, col] += c
                for (j, k), c in Y_c:
                    V[j * p + k, col] += c
                
                for block_idx, sign in C_targets:
                    if block_idx == 0:
                        i_C, k_C = 0, z_idx
                    elif block_idx == 1:
                        i_C, k_C = 0, 3 + z_idx
                    elif block_idx == 2:
                        i_C, k_C = 1, z_idx
                    elif block_idx == 3:
                        i_C, k_C = 1, 3 + z_idx
                    W[k_C * n + i_C, col] += sign
                
                col += 1
        
        rank = 42
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
