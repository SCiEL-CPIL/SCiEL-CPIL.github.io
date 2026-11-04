import numpy as np
import time

def construct():
    n, m, p = 2, 3, 10
    start_time = time.time()
    deadline = 145  # safety margin before 150s kill
    
    # Build the rank-55 Strassen decomposition
    rank = 55
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    # Strassen coefficients for 2x2x2
    # L[s] maps to (i,j) pairs in A (2x2)
    # R[s] maps to (j,k) pairs in B (2x2)  
    # Wc[s] maps to (i,k) pairs in C (2x2)
    L = [
        {0: 1, 3: 1},    # (0,0) and (1,1)
        {2: 1, 3: 1},    # (1,0) and (1,1)
        {0: 1},           # (0,0)
        {3: 1},           # (1,1)
        {0: 1, 1: 1},    # (0,0) and (0,1)
        {2: 1, 0: -1},   # (1,0) and (0,0)
        {1: 1, 3: -1}    # (0,1) and (1,1)
    ]
    R = [
        {0: 1, 3: 1},    # (0,0) and (1,1)
        {0: 1},           # (0,0)
        {1: 1, 3: -1},   # (0,1) and (1,1)
        {2: 1, 0: -1},   # (1,0) and (0,0)
        {3: 1},           # (1,1)
        {0: 1, 1: 1},    # (0,0) and (0,1)
        {2: 1, 3: 1}     # (1,0) and (1,1)
    ]
    Wc = [
        {0: 1, 3: 1},    # (0,0) and (1,1)
        {2: 1, 3: -1},   # (1,0) and (1,1)
        {1: 1, 3: 1},    # (0,1) and (1,1)
        {0: 1, 2: 1},    # (0,0) and (1,0)
        {1: 1, 0: -1},   # (0,1) and (0,0)
        {3: 1},           # (1,1)
        {0: 1}            # (0,0)
    ]
    
    # 5 Strassen blocks covering 2x2x2 subproblems
    # Block structure: rows 0-1 of A, cols 2b-2b+1 of B, outputs 0-1
    for block in range(5):
        k_off = block * 2
        for s in range(7):
            for li, val in L[s].items():
                i, j = li // 2, li % 2
                U[i * m + j][col] = val
            for li, val in R[s].items():
                j, k = li // 2, li % 2
                V[j * p + k_off + k][col] = val
            for li, val in Wc[s].items():
                i, k = li // 2, li % 2
                W[(k_off + k) * n + i][col] = val
            col += 1
    
    # Schoolbook for row 2 of A (the 3rd row)
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Build the target tensor for validation
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try to find redundant columns to reduce rank
    current_rank = rank
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # For each column, check if it's redundant
    for r in range(current_rank):
        if time.time() > deadline:
            break
        
        # Compute contribution of column r
        contrib = np.einsum('a,b,c->abc', U_cur[:, r], V_cur[:, r], W_cur[:, r])
        
        # Check if removing column r still gives valid decomposition
        # T - contrib should be decomposable by remaining columns
        mask = np.ones(current_rank, dtype=bool)
        mask[r] = False
        U_rem = U_cur[:, mask]
        V_rem = V_cur[:, mask]
        W_rem = W_cur[:, mask]
        
        T_rem = np.einsum('ar,br,cr->abc', U_rem, V_rem, W_rem)
        
        if np.allclose(T_rem, T - contrib, atol=1e-9):
            # Column r is redundant, remove it
            U_cur = np.delete(U_cur, r, axis=1)
            V_cur = np.delete(V_cur, r, axis=1)
            W_cur = np.delete(W_cur, r, axis=1)
            current_rank -= 1
            r = -1  # Restart from beginning
            if time.time() > deadline:
                break
    
    # Final validation
    T_final = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
    assert np.allclose(T_final, T, atol=1e-9), "Decomposition is invalid"
    
    # Convert to lists with integer/half-integer entries
    U_out = U_cur.tolist()
    V_out = V_cur.tolist()
    W_out = W_cur.tolist()
    
    for mat in [U_out, V_out, W_out]:
        for row in mat:
            for idx, val in enumerate(row):
                row[idx] = round(val * 2) / 2
    
    return {"U": U_out, "V": V_out, "W": W_out}
