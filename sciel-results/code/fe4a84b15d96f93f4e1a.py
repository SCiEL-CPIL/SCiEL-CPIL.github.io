import time
import numpy as np
from itertools import product
from scipy.optimize import minimize, LinearConstraint

DIMENSIONS = [2, 4, 5]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_nuclear_norm_relaxation(n, m, p, max_rank, time_limit=100):
    """
    Try to find a lower rank decomposition using nuclear norm relaxation.
    This is a heuristic approach that may not always succeed.
    """
    start_time = time.time()
    
    # Build the tensor T[i,j,k] = 1 for all i,j,k
    # We want to minimize the nuclear norm of the unfolded tensor
    # subject to the constraint that the decomposition is exact
    
    # For now, let's try a simpler approach: start with the Strassen solution
    # and try to reduce the rank by finding linear dependencies
    
    strassen_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1], # M7
    ]
    strassen_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [-1,0,1,0], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1],  # M7
    ]
    strassen_W = [
        [1,0,0,1],  # M1: C00+=1, C11+=1
        [0,1,0,-1], # M2: C10+=1, C11+=-1
        [0,0,1,1],  # M3: C01+=1, C11+=1
        [1,1,0,0],  # M4: C00+=1, C10+=1
        [-1,0,1,0], # M5: C00+=-1, C01+=1
        [0,0,0,1],  # M6: C11+=1
        [1,0,0,0],  # M7: C00+=1
    ]
    
    # Total rank: 4 sub-problems * 7 + 8 (schoolbook for k=4) = 36
    base_rank = 4 * 7 + 8
    U_base = np.zeros((n * m, base_rank))
    V_base = np.zeros((m * p, base_rank))
    W_base = np.zeros((p * n, base_rank))
    
    col = 0
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U_base[gi * m + gj, col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V_base[gj * p + gk, col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W_base[gk * n + gi, col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    for i in range(n):
        for j in range(m):
            U_base[i * m + j, col] = 1
            V_base[j * p + 4, col] = 1
            W_base[4 * n + i, col] = 1
            col += 1
    
    # Now try to find a lower rank decomposition
    # We'll use a greedy approach: try to merge columns that are "similar"
    
    best_U = U_base.copy()
    best_V = V_base.copy()
    best_W = W_base.copy()
    best_rank = base_rank
    
    # Try different target ranks
    for target_rank in range(max(20, base_rank - 10), base_rank):
        if time.time() - start_time > time_limit - 20:
            break
        
        # Try to find a rank-target_rank decomposition
        # This is a complex optimization problem, so we'll use a heuristic
        
        # One approach: use SVD on the tensor unfoldings
        # T_{(1)}: (n*m) x (m*p*p)
        # T_{(2)}: (m*p) x (n*m*p)
        # T_{(3)}: (p*n) x (n*m*p)
        
        # For now, let's just try to remove columns that are linearly dependent
        # in a way that preserves the tensor
        
        U_curr = U_base.copy()
        V_curr = V_base.copy()
        W_curr = W_base.copy()
        
        # Try to remove columns one by one
        columns_to_remove = []
        for c in range(base_rank):
            if len(columns_to_remove) >= base_rank - target_rank:
                break
            
            # Check if removing column c still gives a valid decomposition
            # This is expensive, so we'll use a heuristic
            # Check if the column is "redundant" by seeing if it can be
            # expressed as a linear combination of other columns
            
            # For simplicity, we'll just try removing columns and checking
            # if the decomposition still holds (this is O(n*m*p*rank^2))
            
            # Quick check: see if the column has small norm
            col_norm = np.linalg.norm(U_curr[:, c]) + np.linalg.norm(V_curr[:, c]) + np.linalg.norm(W_curr[:, c])
            if col_norm < 0.1:
                columns_to_remove.append(c)
        
        # If we found some columns to remove, try them
        if columns_to_remove:
            # Create new matrices without those columns
            keep_cols = [c for c in range(base_rank) if c not in columns_to_remove]
            U_new = U_curr[:, keep_cols]
            V_new = V_curr[:, keep_cols]
            W_new = W_curr[:, keep_cols]
            
            # Verify
            if _verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist(), n, m, p):
                best_U = U_new
                best_V = V_new
                best_W = W_new
                best_rank = len(keep_cols)
    
    # Convert back to lists
    if best_rank < base_rank:
        U_list = [[0] * best_rank for _ in range(n * m)]
        V_list = [[0] * best_rank for _ in range(m * p)]
        W_list = [[0] * best_rank for _ in range(p * n)]
        
        for i in range(n * m):
            for c in range(best_rank):
                U_list[i][c] = best_U[i, c]
        for i in range(m * p):
            for c in range(best_rank):
                V_list[i][c] = best_V[i, c]
        for i in range(p * n):
            for c in range(best_rank):
                W_list[i][c] = best_W[i, c]
        
        return U_list, V_list, W_list, best_rank
    
    return None

def construct():
    n, m, p = 2, 4, 5
    start_time = time.time()
    
    # Strassen 2x2x2 coefficients (U, V, W) for local indices
    # Local: i,j,k in {0,1}
    # U: [i*2+j] -> 4 entries
    # V: [j*2+k] -> 4 entries
    # W: [k*2+i] -> 4 entries
    
    strassen_U = [
        [1,0,0,1],  # M1
        [0,0,1,1],  # M2
        [1,0,0,0],  # M3
        [0,0,0,1],  # M4
        [1,1,0,0],  # M5
        [-1,0,1,0], # M6
        [0,1,0,-1], # M7
    ]
    strassen_V = [
        [1,0,0,1],  # M1
        [1,0,0,0],  # M2
        [0,1,0,-1], # M3
        [-1,0,1,0], # M4
        [0,0,0,1],  # M5
        [1,1,0,0],  # M6
        [0,0,1,1],  # M7
    ]
    strassen_W = [
        [1,0,0,1],  # M1: C00+=1, C11+=1
        [0,1,0,-1], # M2: C10+=1, C11+=-1
        [0,0,1,1],  # M3: C01+=1, C11+=1
        [1,1,0,0],  # M4: C00+=1, C10+=1
        [-1,0,1,0], # M5: C00+=-1, C01+=1
        [0,0,0,1],  # M6: C11+=1
        [1,0,0,0],  # M7: C00+=1
    ]
    
    # Total rank: 4 sub-problems * 7 + 8 (schoolbook for k=4) = 36
    rank = 4 * 7 + 8
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 20 x rank
    W = [[0] * rank for _ in range(p * n)]  # 10 x rank
    
    col = 0
    # Four 2x2x2 Strassen sub-problems
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                col += 1
    
    # Schoolbook for k=4: 8 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    # Verify the base solution
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank using nuclear norm relaxation
    # This is a heuristic approach that may not always succeed
    time_remaining = 150 - (time.time() - start_time)
    if time_remaining > 20:
        result = _try_nuclear_norm_relaxation(n, m, p, 30, time_limit=time_remaining - 10)
        if result is not None:
            U_new, V_new, W_new, new_rank = result
            if new_rank < rank:
                U, V, W, rank = U_new, V_new, W_new, new_rank
    
    return {"U": U, "V": V, "W": W}
