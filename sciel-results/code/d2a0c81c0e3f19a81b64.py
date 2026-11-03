import numpy as np
import time
import random

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
    # A linear dependency: sum_l alpha_l * (U[:,l] ⊗ V[:,l] ⊗ W[:,l]) = 0
    # This is equivalent to: for all (i,j,k), sum_l alpha_l * U[i,l]*V[j,l]*W[k,l] = 0
    
    # Strategy: sample random points in the tensor space to build a linear system
    # For each sampled (i,j,k), the equation is: sum_l alpha_l * U[i*m+j,l]*V[j*p+k,l]*W[k*n+i,l] = 0
    # We want non-trivial alpha vectors. If we find one, we can eliminate a column.
    
    nU = n * m  # 8
    nV = m * p  # 24
    nW = p * n  # 12
    
    # Build the tensor evaluation matrix: rows = sampled (i,j,k) triples, cols = rank
    # T_eval[row, l] = U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    # Use all 8*24*12 = 2304 possible (i,j,k) combinations
    # This gives a 2304 x 42 matrix. Find its null space.
    
    rows = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = np.array([U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] for l in range(rank)])
                rows.append(row)
    
    T_eval = np.array(rows)  # shape (2304, 42)
    
    # Find the null space of T_eval
    # Use SVD: null space vectors are right singular vectors corresponding to ~0 singular values
    try:
        U_svd, s, Vt = np.linalg.svd(T_eval, full_matrices=False)
        # Tolerance for zero singular values
        tol = 1e-10
        null_space = Vt[s < tol]  # shape (null_dim, 42)
        null_dim = null_space.shape[0]
    except np.linalg.LinAlgError:
        null_dim = 0
        null_space = np.zeros((0, rank))
    
    if null_dim > 0 and time.time() < deadline:
        # We have null_dim linear dependencies. For each, we can eliminate one column.
        # Strategy: for each null vector, find the column with largest |alpha| and eliminate it.
        # But we need to be careful: eliminating a column means expressing it as a combination
        # of others, which changes the decomposition.
        
        # Actually, a cleaner approach: if T_eval has rank r < 42, then the tensor can be
        # decomposed with rank r. We need to find an explicit rank-r decomposition.
        
        # Use the SVD: T_eval = U_svd * diag(s) * Vt
        # The top r left singular vectors span the column space.
        # We can construct a new decomposition using the r independent columns.
        
        # Find which columns of the original decomposition are independent.
        # Use the null space: a set of columns is independent if no null vector has
        # all non-zero entries on them.
        
        # Simpler approach: greedily select independent columns.
        selected = []
        remaining = list(range(rank))
        
        for _ in range(null_dim):
            # Find a null vector and eliminate the column with max |alpha|
            best_null = None
            best_max = 0
            for nv in null_space:
                max_abs = np.max(np.abs(nv))
                if max_abs > best_max:
                    best_max = max_abs
                    best_null = nv
            
            if best_null is None:
                break
            
            # Find column with largest |alpha|
            elim_idx = int(np.argmax(np.abs(best_null)))
            
            # Remove this column and adjust others
            # alpha_l * col_l + sum_{l != elim} alpha_l * col_l = 0
            # col_elim = -sum_{l != elim} (alpha_l / alpha_elim) * col_l
            
            alpha_elim = best_null[elim_idx]
            if abs(alpha_elim) < 1e-15:
                continue
            
            # Update U, V, W: express column elim_idx as combination of others
            for l in range(rank):
                if l == elim_idx:
                    continue
                coeff = -best_null[l] / alpha_elim
                U[:, l] += coeff * U[:, elim_idx]
                V[:, l] += coeff * V[:, elim_idx]
                W[:, l] += coeff * W[:, elim_idx]
            
            # Remove column elim_idx
            mask = [i for i in range(rank) if i != elim_idx]
            U = U[:, mask]
            V = V[:, mask]
            W = W[:, mask]
            rank -= 1
            
            # Recompute null space
            if rank <= 0:
                break
            
            rows = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = np.array([U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] for l in range(rank)])
                        rows.append(row)
            T_eval = np.array(rows)
            
            try:
                U_svd, s, Vt = np.linalg.svd(T_eval, full_matrices=False)
                null_space = Vt[s < tol]
            except np.linalg.LinAlgError:
                break
            
            if null_space.shape[0] == 0:
                break
            
            if time.time() > deadline:
                break
    
    # Verify the decomposition is correct
    # Check that the tensor matches the all-ones tensor
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] for l in range(rank))
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    # If verification fails, fall back to original rank-42
    if max_err > 1e-8:
        # Rebuild original
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
